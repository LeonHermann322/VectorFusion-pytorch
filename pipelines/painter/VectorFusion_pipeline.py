# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import datetime
import wandb
from methods.painter.vectorfusion.utils import TimeLogger
from functools import partial
from PIL import Image
from typing import Literal, Union, AnyStr, List
from skimage.color import rgb2gray
from math import ceil
from PIL import Image

from torchvision.datasets.folder import is_image_file
from torchvision.transforms import ToPILImage
from omegaconf.listconfig import ListConfig
import diffusers
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

import clip
from torchvision import transforms

from diffusers.utils.torch_utils import randn_tensor

from DiffSketcher.libs.metric.lpips_origin.lpips import LPIPS
from DiffSketcher.libs.metric.clip_score.openaiCLIP_loss import CLIPScoreWrapper

from libs.engine import ModelState
from methods.painter.vectorfusion import (
    LSDSPipeline,
    LSDSSDXLPipeline,
    Painter,
    PainterOptimizer,
)
from methods.painter.vectorfusion import (
    channel_saturation_penalty_loss as pixel_penalty_loss,
)
from methods.painter.vectorfusion import xing_loss_fn
from methods.painter.vectorfusion.utils import log_tensor_img, plt_batch, view_images
from methods.diffusers_warp import init_diffusion_pipeline, model2res
from diffusers import StableDiffusionPipeline
from methods.diffvg_warp import init_diffvg
from DiffSketcher.methods.token2attn.attn_control import AttentionStore, EmptyControl

from create_gif import create_gif_from_svgs


def get_clip_score(prompt, img, device):  # local import
    print("todo: use batch")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    text_input = clip.tokenize([prompt]).to(device)
    text_features = clip_model.encode_text(text_input)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    clip_img = preprocess(img).unsqueeze(0).to(device)
    image_features = clip_model.encode_image(clip_img)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # clip score
    similarity_score = (text_features @ image_features.T).squeeze(0)

    return similarity_score[0].item()


class VectorFusionPipeline(ModelState):
    def __init__(self, args, eval_mode=False):
        now = datetime.datetime.now()

        # Format the date and time as a string, for example '2024-01-06_15-30-25'
        formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        self.eval_mode = eval_mode
        # Your base file path or directory

        logdir_ = (
            f"{'scratch' if args.skip_live else 'baseline'}"
            f"-{args.model_id}"
            f"-{args.style}"
            f"-sd{args.seed}"
            f"-im{args.image_size}"
            f"-P{args.num_paths}"
            f"{'-RePath' if args.path_reinit.use else ''}"
            # f"{formatted_date_time}"
        )

        if eval_mode:
            logdir_ = "eval_generation"
        super().__init__(args, log_path_suffix=logdir_)

        assert args.style in ["iconography", "pixelart", "sketch"]

        # create log dir
        self.png_logs_dir = self.results_path / "png_logs"
        self.svg_logs_dir = self.results_path / "svg_logs"
        self.ft_png_logs_dir = self.results_path / "ft_png_logs"
        self.ft_svg_logs_dir = self.results_path / "ft_svg_logs"
        self.sd_sample_dir = self.results_path / "sd_samples"
        self.reinit_dir = self.results_path / "reinit_logs"

        if self.accelerator.is_main_process:
            self.png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.svg_logs_dir.mkdir(parents=True, exist_ok=True)
            self.ft_png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.ft_svg_logs_dir.mkdir(parents=True, exist_ok=True)
            self.sd_sample_dir.mkdir(parents=True, exist_ok=True)
            self.reinit_dir.mkdir(parents=True, exist_ok=True)

        self.select_fpth = self.results_path / "select_sample.png"

        init_diffvg(self.device, True, args.print_timing)

        if args.model_id == "sdxl":
            # default LSDSSDXLPipeline scheduler is EulerDiscreteScheduler
            # when LSDSSDXLPipeline calls, scheduler.timesteps will change in step 4
            # which causes problem in sds add_noise() function
            # because the random t may not in scheduler.timesteps
            custom_pipeline = LSDSSDXLPipeline
            custom_scheduler = diffusers.DPMSolverMultistepScheduler
        elif args.model_id == "sd21":
            custom_pipeline = LSDSPipeline
            custom_scheduler = diffusers.DDIMScheduler
        else:  # sd14, sd15
            custom_pipeline = LSDSPipeline
            custom_scheduler = diffusers.PNDMScheduler

        self.diffusion: StableDiffusionPipeline = init_diffusion_pipeline(
            args.model_id,
            custom_pipeline=custom_pipeline,
            custom_scheduler=custom_scheduler,
            device=self.device,
            local_files_only=not args.download,
            force_download=args.force_download,
            resume_download=args.resume_download,
            ldm_speed_up=args.ldm_speed_up,
            enable_xformers=args.enable_xformers,
            gradient_checkpoint=args.gradient_checkpoint,
            lora_path=args.lora_path,
        )

        self.clip_score_fn = CLIPScoreWrapper(
            self.args.clip.model_name,
            device=self.device,
            visual_score=True,
            feats_loss_type=self.args.clip.feats_loss_type,
            feats_loss_weights=self.args.clip.feats_loss_weights,
            fc_loss_weight=self.args.clip.fc_loss_weight,
        )
        self.lpips_loss_fn = LPIPS(net=self.args.perceptual.lpips_net).to(self.device)

        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)

        if args.style == "pixelart":
            args.path_schedule = "list"
            args.schedule_each = list([args.grid])

        if args.train_stroke:
            args.path_reinit.use = False
            self.print("-> train stroke: True, then disable reinitialize_paths.")

    # Taken from https://github.com/ximinng/DiffSketcher/blob/e4c03a6abd30dcb4b63ae5867f0bcc181ad0dccc/pipelines/painter/diffsketcher_pipeline.py
    @property
    def clip_norm_(self):
        return transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

    def clip_pair_augment(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        im_res: int,
        augments: str = "affine_norm",
        num_aug: int = 4,
    ):
        # init augmentations
        augment_list = []
        if "affine" in augments:
            augment_list.append(
                transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5)
            )
            augment_list.append(
                transforms.RandomResizedCrop(im_res, scale=(0.8, 0.8), ratio=(1.0, 1.0))
            )
        augment_list.append(self.clip_norm_)  # CLIP Normalize

        # compose augmentations
        augment_compose = transforms.Compose(augment_list)
        # make augmentation pairs
        x_augs, y_augs = [self.clip_score_fn.normalize(x)], [
            self.clip_score_fn.normalize(y)
        ]
        # repeat N times
        for n in range(num_aug):
            augmented_pair = augment_compose(torch.cat([x, y]))
            x_augs.append(augmented_pair[0].unsqueeze(0))
            y_augs.append(augmented_pair[1].unsqueeze(0))
        xs = torch.cat(x_augs, dim=0)
        ys = torch.cat(y_augs, dim=0)
        return xs, ys

    def get_path_schedule(self, schedule_each: Union[int, List]):
        if self.args.path_schedule == "repeat":
            return int(self.args.num_paths / schedule_each) * [schedule_each]
        elif self.args.path_schedule == "list":
            assert isinstance(self.args.schedule_each, list) or isinstance(
                self.args.schedule_each, ListConfig
            )
            return schedule_each
        else:
            raise NotImplementedError

    def target_file_preprocess(self, tar_path: AnyStr):
        process_comp = transforms.Compose(
            [
                transforms.Resize(size=(self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.unsqueeze(0)),
            ]
        )

        tar_pil = Image.open(tar_path).convert("RGB")  # open file
        target_img = process_comp(tar_pil)  # preprocess
        target_img = target_img.to(self.device)
        return target_img

    @torch.no_grad()
    def rejection_sampling(
        self, img_caption: Union[AnyStr, List], diffusion_samples: List
    ):  # local import

        clip_model, preprocess = clip.load("ViT-B/32", device=self.device)

        text_input = clip.tokenize([img_caption]).to(self.device)
        text_features = clip_model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        clip_images = torch.stack(
            [preprocess(sample) for sample in diffusion_samples]
        ).to(self.device)
        image_features = clip_model.encode_image(clip_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # clip score
        similarity_scores = (text_features @ image_features.T).squeeze(0)

        selected_image_index = similarity_scores.argmax().item()
        selected_image = diffusion_samples[selected_image_index]
        return selected_image, selected_image_index

    def get_clip_text_embeddings(
        self, text_prompt: str, attribute: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_emb = self.clip_score_fn.encode_text(text_prompt).cuda()
        attribute_emb = self.clip_score_fn.encode_text(attribute).cuda()
        return prompt_emb, attribute_emb

    def get_diffusion_text_embeddings(
        self, text_prompt: str, attribute: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_emb = self.diffusion._encode_prompt(
            text_prompt,
            num_images_per_prompt=1,
            device=self.diffusion._execution_device,
            do_classifier_free_guidance=self.args.guidance_scale > 1.0,
        ).cuda()
        attribute_emb = self.diffusion._encode_prompt(
            attribute,
            num_images_per_prompt=1,
            device=self.diffusion._execution_device,
            do_classifier_free_guidance=self.args.guidance_scale > 1.0,
        ).cuda()
        return prompt_emb, attribute_emb

    def sample_diffusion_model(
        self,
        input: torch.Tensor | str,
        inference_steps: int,
        latents: torch.Tensor | None = None,
        return_type: Literal["pil", "latents", "pil+latents"] | None = None,
        return_dict: bool = False,
        seed=None,
    ):
        height = width = model2res(self.args.model_id)
        if type(input) is str:
            return self.diffusion(
                prompt=input,
                negative_prompt=self.args.negative_prompt,
                height=height,
                width=width,
                controller=EmptyControl(),
                latents=latents,
                num_images_per_prompt=1,
                num_inference_steps=inference_steps,
                guidance_scale=self.args.guidance_scale,
                output_type=return_type,
                return_dict=return_dict,
                generator=self.g_device,
            )
        else:
            return self.diffusion(
                "",
                embedding=input,
                height=height,
                width=width,
                controller=EmptyControl(),
                latents=latents,
                num_images_per_prompt=1,
                num_inference_steps=inference_steps,
                guidance_scale=self.args.guidance_scale,
                output_type=return_type,
                return_dict=return_dict,
                generator=self.g_device,
            )

    def get_clip_image_embedding(self, image):
        self.clip_score_fn = self.clip_score_fn.cpu()
        preprocess_img = self.clip_score_fn.preprocess(image).unsqueeze(0)
        encoded_img = self.clip_score_fn.encode_image(preprocess_img)
        return encoded_img

    def latent_initialization(self, dtype):
        height = width = model2res(self.args.model_id)
        try:
            num_channels_latents = self.diffusion.unet.config.in_channels
        except Exception or Warning:
            num_channels_latents = self.diffusion.unet.in_channels
        latents_shape = (
            1,
            num_channels_latents,
            height // self.diffusion.vae_scale_factor,
            width // self.diffusion.vae_scale_factor,
        )
        latents = randn_tensor(
            shape=latents_shape, device=self.diffusion._execution_device, dtype=dtype
        )
        return latents

    def diffusion_sampling(self, text_prompt: AnyStr, attribute: AnyStr | None = None):
        """sampling K images"""
        diffusion_samples = []
        for i in range(self.args.K):
            file_path = self.sd_sample_dir / f"samples_{i}.png"
            if self.args.reverse_diffusion_mode == "sampling":
                if self.args.reverse_diffusion_embedding_mode == "hardcut":
                    inf_steps_prompt = ceil(
                        self.args.num_inference_steps * self.args.attribute_importance
                    )
                    inf_steps_attr = self.args.num_inference_steps - inf_steps_prompt

                    latents, _ = self.sample_diffusion_model(
                        input=text_prompt,
                        inference_steps=inf_steps_prompt,
                        return_type="latents",
                    )
                    outputs = self.sample_diffusion_model(
                        input=attribute,
                        inference_steps=inf_steps_attr,
                        latents=latents,
                        return_type="pil",
                        return_dict=True,
                    )
                else:
                    # Encoding text input for diffusion model and CLIP loss
                    (
                        prompt_emb_diffusion,
                        attribute_emb_diffusion,
                    ) = self.get_diffusion_text_embeddings(text_prompt, attribute)
                    (
                        prompt_emb_clip,
                        attribute_emb_clip,
                    ) = self.get_clip_text_embeddings(text_prompt, attribute)

                    init_latents = self.latent_initialization(
                        prompt_emb_diffusion.dtype
                    )
                    # Creation with only attribute
                    outputs = self.sample_diffusion_model(
                        input=attribute,
                        inference_steps=self.args.num_inference_steps,
                        latents=init_latents,
                        return_type="pil",
                        return_dict=True,
                    )
                    outputs_np = [np.array(img) for img in outputs.images]
                    view_images(
                        outputs_np,
                        save_image=True,
                        fp=self.sd_sample_dir / f"samples_{i}_attr.png",
                    )
                    # Original creation without attribute
                    outputs = self.sample_diffusion_model(
                        input=text_prompt,
                        inference_steps=self.args.num_inference_steps,
                        latents=init_latents,
                        return_type="pil",
                        return_dict=True,
                    )
                    outputs_np = [np.array(img) for img in outputs.images]
                    view_images(
                        outputs_np,
                        save_image=True,
                        fp=self.sd_sample_dir / f"samples_{i}_orig.png",
                    )

                    # Encode sample of prompt with clip
                    self.clip_score_fn = self.clip_score_fn.cuda()
                    orig_img_pil = outputs.images[0]

                    enc_orig_img = self.clip_score_fn.encode_image(
                        self.clip_score_fn.preprocess(
                            # torch.from_numpy(np.array(sample)).cpu()
                            orig_img_pil
                        )
                        .cuda()
                        .unsqueeze(0)
                    )

                    # Optimize lambda_coeff for:
                    #   1. Semantic similarity compared to original image (Perceptual loss)
                    #   2. Style conformity of image in respect to attribute embedding (CLIP directional loss)
                    losses = []
                    outputs_array = []
                    for step in range(1, int(1.0 / self.args.lambda_step_size)):
                        lambda_coeff = torch.tensor(
                            data=[self.args.lambda_step_size * step]
                        ).cuda()
                        inp_emb = (
                            (1.0 - lambda_coeff) * prompt_emb_diffusion
                            + lambda_coeff * attribute_emb_diffusion
                        )
                        outputs = self.sample_diffusion_model(
                            input=inp_emb,
                            inference_steps=self.args.num_inference_steps,
                            latents=init_latents,
                            return_type="pil",
                            return_dict=True,
                        )
                        img = outputs.images[0]

                        # Encoding image with CLIP
                        finetune_img = (
                            self.clip_score_fn.preprocess(img).cuda().unsqueeze(0)
                        )
                        enc_finetune_img = self.clip_score_fn.encode_image(finetune_img)

                        # Checking semantic change in the image
                        perceptual_loss_fn = partial(
                            self.lpips_loss_fn.forward,
                            return_per_layer=False,
                            normalize=False,
                        )
                        l_perceptual = perceptual_loss_fn(
                            torch.from_numpy(np.array(orig_img_pil))
                            .cuda()
                            .permute(2, 0, 1)
                            .unsqueeze(0),
                            torch.from_numpy(np.array(img))
                            .cuda()
                            .permute(2, 0, 1)
                            .unsqueeze(0),
                        ).mean()

                        # Directional loss with current denoised image
                        l_clip_directional = self.clip_score_fn.directional_loss(
                            src_text=prompt_emb_clip,
                            src_img=enc_orig_img,
                            tar_text=attribute_emb_clip,
                            tar_img=enc_finetune_img,
                        )

                        # Update lambda_coeff
                        loss = l_clip_directional + self.args.beta * l_perceptual

                        print(
                            f"{i} Iteration: lambda = {lambda_coeff.item()}, loss = {loss.item()}, clip directional = {l_clip_directional.item()}, perceptual = {l_perceptual}"
                        )

                        losses.append(loss.item())
                        outputs_array.append(outputs)

                    min_loss_idx = np.argmin(losses)

                    for lambda_idx, output in enumerate(outputs_array):
                        file_path = (
                            self.sd_sample_dir
                            / f"samples_{i}_{(lambda_idx + 1) * self.args.lambda_step_size}.png"
                        )
                        outputs_np = [np.array(img) for img in output.images]
                        view_images(outputs_np, save_image=True, fp=file_path)

                    file_path = (
                        self.sd_sample_dir
                        / f"samples_{i}_optim_{(min_loss_idx + 1) * self.args.lambda_step_size}.png"
                    )
                    outputs = outputs_array[min_loss_idx]
            else:
                outputs = self.sample_diffusion_model(
                    input=text_prompt,
                    inference_steps=self.args.num_inference_steps,
                    return_type="pil",
                    return_dict=True,
                )
            outputs_np = [np.array(img) for img in outputs.images]
            view_images(outputs_np, save_image=True, fp=file_path)
            diffusion_samples.extend(outputs.images)

        self.print(
            f"num_generated_samples: {len(diffusion_samples)}, shape: {outputs_np[0].shape}"
        )

        return diffusion_samples

    def LIVE_rendering(self, text_prompt: AnyStr, attribute: AnyStr):
        select_fpth = self.select_fpth
        # sampling K images
        k_sampling_timelog = TimeLogger(
            "k_sampling_and_clip_rejction", self.args.use_wandb
        )
        if self.args.reverse_diffusion_mode == "sampling":
            diffusion_samples = self.diffusion_sampling(text_prompt, attribute)
        else:
            diffusion_samples = self.diffusion_sampling(text_prompt)
        # rejection sampling
        select_target, _ = self.rejection_sampling(
            (
                attribute
                if self.args.reverse_diffusion_mode == "sampling"
                else text_prompt
            ),
            diffusion_samples,
        )
        select_target_pil = Image.fromarray(np.asarray(select_target))  # numpy to PIL
        select_target_pil.save(select_fpth)

        k_sampling_timelog.finish()

        live_timelog = TimeLogger("live", self.args.use_wandb)

        # empty cache
        torch.cuda.empty_cache()

        # load target file
        assert select_fpth.exists(), f"{select_fpth} is not exist!"
        target_img = self.target_file_preprocess(select_fpth.as_posix())

        self.print(f"load target file from: {select_fpth.as_posix()}")

        # log path_schedule
        path_schedule = self.get_path_schedule(self.args.schedule_each)
        self.print(f"path_schedule: {path_schedule}")

        renderer = self.load_renderer(target_img)
        # first init center
        renderer.component_wise_path_init(pred=None, init_type=self.args.coord_init)

        optimizer_list = [
            PainterOptimizer(
                renderer,
                self.args.style,
                self.args.num_iter,
                self.args.lr_base,
                self.args.train_stroke,
                self.args.trainable_bg,
            )
            for _ in range(len(path_schedule))
        ]

        pathn_record = []
        loss_weight_keep = 0

        total_step = len(path_schedule) * self.args.num_iter
        with tqdm(
            initial=self.step,
            total=total_step,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            for path_idx, pathn in enumerate(path_schedule):
                # record path
                pathn_record.append(pathn)
                # init graphic
                img = renderer.init_image(stage=0, num_paths=pathn)
                log_tensor_img(
                    img, self.results_path, output_prefix=f"init_img_{path_idx}"
                )
                # rebuild optimizer
                optimizer_list[path_idx].init_optimizers(
                    pid_delta=int(path_idx * pathn)
                )

                pbar.write(
                    f"=> adding {pathn} paths, n_path: {sum(pathn_record)}, "
                    f"n_points: {len(renderer.get_point_parameters())}, "
                    f"n_colors: {len(renderer.get_color_parameters())}"
                )

                for t in range(self.args.num_iter):
                    raster_img = renderer.get_image(step=t).to(self.device)

                    if self.args.use_distance_weighted_loss and not (
                        self.args.style == "pixelart"
                    ):
                        loss_weight = renderer.calc_distance_weight(loss_weight_keep)

                    # reconstruction loss
                    if self.args.style == "pixelart":
                        loss_recon = torch.nn.functional.l1_loss(raster_img, target_img)
                    else:  # UDF loss
                        loss_recon = (raster_img - target_img) ** 2
                        loss_recon = (loss_recon.sum(1) * loss_weight).mean()

                    # Xing Loss for Self-Interaction Problem
                    loss_xing = torch.tensor(0.0)
                    if self.args.style == "iconography":
                        loss_xing = (
                            xing_loss_fn(renderer.get_point_parameters())
                            * self.args.xing_loss_weight
                        )

                    # total loss
                    loss = loss_recon + loss_xing

                    lr_str = ""
                    for k, lr in optimizer_list[path_idx].get_lr().items():
                        lr_str += f"{k}_lr: {lr:.4f}, "

                    pbar.set_description(
                        lr_str + f"L_total: {loss.item():.4f}, "
                        f"L_recon: {loss_recon.item():.4f}, "
                        f"L_xing: {loss_xing.item()}"
                    )

                    # optimization
                    for i in range(path_idx + 1):
                        optimizer_list[i].zero_grad_()

                    loss.backward()

                    for i in range(path_idx + 1):
                        optimizer_list[i].step_()

                    renderer.clip_curve_shape()

                    if self.args.lr_scheduler:
                        for i in range(path_idx + 1):
                            optimizer_list[i].update_lr()

                    if (
                        self.step % self.args.save_step == 0
                        and self.accelerator.is_main_process
                    ):
                        plt_batch(
                            target_img,
                            raster_img,
                            self.step,
                            prompt=text_prompt,
                            save_path=self.png_logs_dir.as_posix(),
                            name=f"iter{self.step}",
                        )
                        renderer.save_svg(
                            self.svg_logs_dir / f"svg_iter{self.step}.svg"
                        )

                    self.step += 1
                    pbar.update(1)

                # end a set of path optimization
                if self.args.use_distance_weighted_loss and not (
                    self.args.style == "pixelart"
                ):
                    loss_weight_keep = loss_weight.detach().cpu().numpy() * 1
                # recalculate the coordinates for the new join path
                renderer.component_wise_path_init(raster_img)

        # end LIVE
        final_svg_fpth = self.results_path / "live_stage_one_final.svg"
        renderer.save_svg(final_svg_fpth)

        live_timelog.finish()

        return target_img, final_svg_fpth

    def paint_for_evaluation(self, text_prompt, test_data_path, image_id):
        self.test_data_path = test_data_path
        self.image_id = image_id

        self.painterly_rendering(text_prompt)

    def painterly_rendering(self, text_prompt: AnyStr, attribute: AnyStr | None = None):
        # log prompts
        self.print(f"prompt: {text_prompt}")
        self.print(f"negative_prompt: {self.args.negative_prompt}\n")

        if self.args.skip_live:
            target_img = torch.zeros(
                self.args.batch_size, 3, self.args.image_size, self.args.image_size
            )
            final_svg_fpth = None
            self.print("from scratch with Score Distillation Sampling...")
        else:
            # text-to-img-to-svg
            if self.args.coord_init == "attention":
                raise ("Attention Init does not work with live,")
            target_img, final_svg_fpth = self.LIVE_rendering(text_prompt, attribute)
            torch.cuda.empty_cache()
            self.args.path_svg = final_svg_fpth
            self.print("\nfine-tune SVG via Score Distillation Samplig...")

        if (
            self.args.reverse_diffusion_mode == "finetuning"
            and self.args.reverse_diffusion_embedding_mode == "interpolated"
        ):
            prompt_embedding = self.diffusion._encode_prompt(
                text_prompt,
                self.device,
                1,
                self.args.sds.guidance_scale > 1.0,
                negative_prompt=self.args.negative_prompt,
            )
            attribute_embedding = self.diffusion._encode_prompt(
                attribute,
                self.device,
                1,
                self.args.sds.guidance_scale > 1.0,
                negative_prompt=self.args.negative_prompt,
            )

        attention_map = None
        if self.args.coord_init == "attention":
            target_img, attention_map = self.extract_ldm_attn(prompts=text_prompt)
            target_img = self.get_target(target_img, self.args.image_size)
            target_img = target_img.detach()  # inputs as GT
            self.print("inputs shape: ", target_img.shape)

        vetor_fusion_timelog = TimeLogger(
            name="vector_fusion_part", use_wandb=self.args.use_wandb
        )
        renderer = self.load_renderer(
            target_img, path_svg=self.args.path_svg, attention_map=attention_map
        )
        target_img.to(self.device)

        if self.args.skip_live:
            init_type = "random"
            if self.args.coord_init == "attention":
                init_type = self.args.coord_init
            renderer.component_wise_path_init(pred=None, init_type=init_type)

        img = renderer.init_image(stage=0, num_paths=self.args.num_paths)

        log_tensor_img(img, self.results_path, output_prefix=f"init_img_stage_two")

        optimizer = PainterOptimizer(
            renderer,
            self.args.style,
            self.args.sds.num_iter,
            self.args.lr_base,
            self.args.train_stroke,
            self.args.trainable_bg,
        )
        optimizer.init_optimizers()

        self.print(f"-> Painter points Params: {len(renderer.get_point_parameters())}")
        self.print(f"-> Painter color Params: {len(renderer.get_color_parameters())}")

        self.step = 0  # reset global step
        total_step = self.args.sds.num_iter
        path_reinit = self.args.path_reinit

        max_attribute_shift_step = total_step * (1.0 - self.args.attribute_importance)

        self.print(f"\ntotal sds optimization steps: {total_step}")
        with tqdm(
            initial=self.step,
            total=total_step,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.step < total_step:
                raster_img = renderer.get_image(step=self.step).to(self.device)

                if self.args.reverse_diffusion_mode == "finetuning":
                    if self.args.reverse_diffusion_embedding_mode == "interpolated":
                        # rev coeff 0.0 -> all prompt, 1.0 -> all attribute
                        rev_coeff = min(
                            (
                                self.step / max_attribute_shift_step
                                if max_attribute_shift_step > 0
                                else 1.0
                            ),
                            1.0,
                        )
                        text_embedding = (
                            rev_coeff * attribute_embedding
                            + (1.0 - rev_coeff) * prompt_embedding
                        )
                        L_sds, grad = self.diffusion.reverse_diffusion_sds(
                            raster_img,
                            im_size=self.args.sds.im_size,
                            text_embedding=text_embedding,
                            guidance_scale=self.args.sds.guidance_scale,
                            grad_scale=self.args.sds.grad_scale,
                            t_range=list(self.args.sds.t_range),
                        )
                    else:
                        shift_to_attribute = (
                            self.step > total_step * self.args.attribute_importance
                        )
                        prompt = attribute if shift_to_attribute else text_prompt

                        L_sds, grad = self.diffusion.score_distillation_sampling(
                            raster_img,
                            im_size=self.args.sds.im_size,
                            prompt=[prompt],
                            negative_prompt=self.args.negative_prompt,
                            guidance_scale=self.args.sds.guidance_scale,
                            grad_scale=self.args.sds.grad_scale,
                            t_range=list(self.args.sds.t_range),
                        )
                elif self.args.reverse_diffusion_mode == "sampling":
                    L_sds, grad = self.diffusion.score_distillation_sampling(
                        raster_img,
                        im_size=self.args.sds.im_size,
                        prompt=[attribute],
                        negative_prompt=self.args.negative_prompt,
                        guidance_scale=self.args.sds.guidance_scale,
                        grad_scale=self.args.sds.grad_scale,
                        t_range=list(self.args.sds.t_range),
                    )
                else:
                    L_sds, grad = self.diffusion.score_distillation_sampling(
                        raster_img,
                        im_size=self.args.sds.im_size,
                        prompt=[text_prompt],
                        negative_prompt=self.args.negative_prompt,
                        guidance_scale=self.args.sds.guidance_scale,
                        grad_scale=self.args.sds.grad_scale,
                        t_range=list(self.args.sds.t_range),
                    )
                # Xing Loss for Self-Interaction Problem
                L_add = torch.tensor(0.0)
                if self.args.style == "iconography":
                    L_add = (
                        xing_loss_fn(renderer.get_point_parameters())
                        * self.args.xing_loss_weight
                    )
                # pixel_penalty_loss to combat oversaturation
                if self.args.style == "pixelart":
                    L_add = pixel_penalty_loss(raster_img) * self.args.penalty_weight

                # Taken from https://github.com/ximinng/DiffSketcher/blob/e4c03a6abd30dcb4b63ae5867f0bcc181ad0dccc/pipelines/painter/diffsketcher_pipeline.py
                l_percep = torch.tensor(0.0)
                total_visual_loss = torch.tensor(0.0)
                if (not self.args.skip_live) or self.args.use_jvsp:
                    # Similarity loss to diffusion sample
                    # L_sim = (
                    #     # self.diffusion.similarity_loss(raster_img, target_img)
                    #     self.diffusion.kl_div(raster_img, target_img)
                    #     * self.args.sim_loss_weight
                    # )

                    perceptual_loss_fn = partial(
                        self.lpips_loss_fn.forward,
                        return_per_layer=False,
                        normalize=False,
                    )
                    target_img = target_img.to(self.device)
                    raster_img = raster_img.to(self.device)
                    l_perceptual = perceptual_loss_fn(raster_img, target_img).mean()
                    l_percep = l_perceptual * self.args.perceptual.coeff

                    # Taken from https://github.com/ximinng/DiffSketcher/blob/e4c03a6abd30dcb4b63ae5867f0bcc181ad0dccc/pipelines/painter/diffsketcher_pipeline.py
                    # CLIP data augmentation
                    raster_sketch_aug, inputs_aug = self.clip_pair_augment(
                        raster_img,
                        target_img,
                        im_res=224,
                        augments=self.args.clip.augmentations,
                        num_aug=self.args.clip.num_aug,
                    )

                    if self.args.clip.vis_loss > 0:
                        (
                            l_clip_fc,
                            l_clip_conv,
                        ) = self.clip_score_fn.compute_visual_distance(
                            raster_sketch_aug, inputs_aug, clip_norm=False
                        )
                        clip_conv_loss_sum = sum(l_clip_conv)
                        total_visual_loss = self.args.clip.vis_loss * (
                            clip_conv_loss_sum + l_clip_fc
                        )

                loss = L_sds + L_add + l_percep + total_visual_loss

                if self.args.use_wandb:
                    wandb.log(
                        {
                            "L_total": loss.item(),
                            "L_sds": L_sds.item(),
                            "L_add": L_add.item(),
                            "l_percep": l_percep.item(),
                            "total_visual_loss": total_visual_loss.item(),
                        }
                    )

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                if self.args.style != "sketch":
                    renderer.clip_curve_shape()

                # re-init paths
                if (
                    self.step % path_reinit.freq == 0
                    and self.step < path_reinit.stop_step
                    and self.step != 0
                ):
                    renderer.reinitialize_paths(
                        path_reinit.use,  # on-off
                        path_reinit.opacity_threshold,
                        path_reinit.area_threshold,
                        fpath=self.reinit_dir / f"reinit-{self.step}.svg",
                    )

                # update lr
                if self.args.lr_scheduler:
                    optimizer.update_lr()

                lr_str = ""
                for k, lr in optimizer.get_lr().items():
                    lr_str += f"{k}_lr: {lr:.4f}, "

                pbar.set_description(
                    lr_str
                    + f"L_total: {loss.item():.4f}, L_add: {L_add.item():.5e}, L_percep: {l_percep.item():.5e}, Total vis loss: {total_visual_loss.item():.5e}, "
                    f"sds: {grad.item():.5e}"
                )

                if (
                    self.step % self.args.save_step == 0
                    and self.accelerator.is_main_process
                ):
                    image_array = plt_batch(
                        target_img,
                        raster_img,
                        self.step,
                        prompt=text_prompt,
                        save_path=self.ft_png_logs_dir.as_posix(),
                        name=f"iter{self.step}",
                    )
                    image = Image.fromarray(image_array)
                    caption = f"step {self.step}"
                    if self.args.log_image and self.args.use_wandb:
                        wandb.log({"image": wandb.Image(image, caption=caption)})
                    renderer.save_svg(self.ft_svg_logs_dir / f"svg_iter{self.step}.svg")
                    # log clip score
                    if self.args.log_clip:
                        final_raster_img = renderer.get_image(self.step)
                        clip_score = get_clip_score(
                            text_prompt,
                            ToPILImage()(final_raster_img.squeeze()),
                            self.device,
                        )
                        print(clip_score)
                        if self.args.use_wandb:
                            wandb.log({"clip_score": clip_score})

                self.step += 1
                pbar.update(1)

        final_svg_fpth = self.results_path / "finetune_final.svg"
        renderer.save_svg(final_svg_fpth)

        final_raster_img = renderer.get_image(self.step)
        clip_score = get_clip_score(
            text_prompt, ToPILImage()(final_raster_img.squeeze()), self.device
        )
        print(clip_score)
        if self.args.use_wandb:
            wandb.log({"clip_score": clip_score})
        vetor_fusion_timelog.finish()

        if self.eval_mode:
            print("saving svg to testdataset")

            final_svg_fpth = self.test_data_path / f"{self.image_id}.svg"
            print(final_svg_fpth)
            renderer.save_svg(final_svg_fpth)

        # create gif of logs
        create_gif_from_svgs(self.ft_svg_logs_dir, self.results_path)

        self.close(msg="painterly rendering complete.")

    def load_renderer(self, target_img, path_svg=None, attention_map=None):
        renderer = Painter(
            self.args.style,
            target_img,
            self.args.num_segments,
            self.args.segment_init,
            self.args.radius,
            self.args.image_size,
            self.args.grid,
            self.args.trainable_bg,
            self.args.train_stroke,
            self.args.width,
            path_svg=path_svg,
            device=self.device,
            attention_map=attention_map,
            softmax_temp=self.args.softmax_temp,
            num_paths=self.args.num_paths,
            num_stages=self.args.num_stages,
            xdog_intersec=self.args.xdog_intersec,
        )
        return renderer

    def extract_ldm_attn(self, prompts):
        print("\n\nextract_ldm_attn with k sampling\n\n")
        # init controller
        controller = (
            AttentionStore() if self.args.coord_init == "attention" else EmptyControl()
        )

        cross_attention_outputs = []
        self_attention_comp_outputs = []

        select_fpth = self.select_fpth

        k_sampling_timelog = TimeLogger(
            "k_sampling_and_clip_rejction", use_wandb=self.args.use_wandb
        )

        # sampling K images
        diffusion_samples = []
        for i in range(self.args.K):
            height = width = model2res(self.args.model_id)
            outputs = self.diffusion(
                prompt=[prompts],
                negative_prompt=self.args.negative_prompt,
                height=height,
                width=width,
                num_images_per_prompt=1,
                num_inference_steps=self.args.num_inference_steps,
                guidance_scale=self.args.guidance_scale,
                generator=self.g_device,
                controller=controller,
            )
            outputs_np = [np.array(img) for img in outputs.images]
            view_images(
                outputs_np, save_image=True, fp=self.sd_sample_dir / f"samples_{i}.png"
            )

            """ldm cross-attention map"""
            cross_attention_maps, tokens = self.diffusion.get_cross_attention(
                [prompts],
                controller,
                res=self.args.cross_attn_res,
                from_where=("up", "down"),
                save_path=self.results_path / f"cross_attn{i}.png",
            )

            """ldm self-attention map"""
            self_attention_maps, svd, vh_ = self.diffusion.get_self_attention_comp(
                [prompts],
                controller,
                res=self.args.self_attn_res,
                from_where=("up", "down"),
                img_size=self.args.image_size,
                max_com=self.args.max_com,
                save_path=self.results_path,
                index=i,
            )

            cross_attention_outputs.append((cross_attention_maps, tokens))
            self_attention_comp_outputs.append((self_attention_maps, svd, vh_))

            diffusion_samples.extend(outputs.images)
            controller.reset()

        self.print(
            f"num_generated_samples: {len(diffusion_samples)}, shape: {outputs_np[0].shape}"
        )
        # rejection sampling
        select_target, target_index = self.rejection_sampling(
            prompts, diffusion_samples
        )
        # empty cache
        torch.cuda.empty_cache()

        select_fpth = self.select_fpth
        select_target_pil = Image.fromarray(np.asarray(select_target))  # numpy to PIL
        select_target_pil.save(select_fpth)

        k_sampling_timelog.finish()
        attention_collect_timelog = TimeLogger(
            "attention_collect", use_wandb=self.args.use_wandb
        )

        target_path = self.results_path / "ldm_generated_image.png"
        view_images([np.array(select_target)], save_image=True, fp=target_path)

        """Attention Map calculating"""
        self_attention_maps, svd, vh_ = self_attention_comp_outputs[target_index]
        cross_attention_maps, tokens = cross_attention_outputs[target_index]

        self.print(
            f"the length of tokens is {len(tokens)}, select {self.args.token_ind}-th token"
        )
        # [res, res, seq_len]
        self.print(f"origin cross_attn_map shape: {cross_attention_maps.shape}")
        # [res, res]
        cross_attn_map = cross_attention_maps[:, :, self.args.token_ind]
        self.print(f"select cross_attn_map shape: {cross_attn_map.shape}\n")
        cross_attn_map = 255 * cross_attn_map / cross_attn_map.max()
        # [res, res, 3]
        cross_attn_map = cross_attn_map.unsqueeze(-1).expand(*cross_attn_map.shape, 3)
        # [3, res, res]
        cross_attn_map = cross_attn_map.permute(2, 0, 1).unsqueeze(0)
        # [3, clip_size, clip_size]
        cross_attn_map = F.interpolate(
            cross_attn_map, size=self.args.image_size, mode="bicubic"
        )
        cross_attn_map = torch.clamp(cross_attn_map, min=0, max=255)
        # rgb to gray
        cross_attn_map = rgb2gray(cross_attn_map.squeeze(0).permute(1, 2, 0)).astype(
            np.float32
        )
        # torch to numpy
        if (
            cross_attn_map.shape[-1] != self.args.image_size
            and cross_attn_map.shape[-2] != self.args.image_size
        ):
            cross_attn_map = cross_attn_map.reshape(
                self.args.image_size, self.args.image_size
            )
        # to [0, 1]
        cross_attn_map = (cross_attn_map - cross_attn_map.min()) / (
            cross_attn_map.max() - cross_attn_map.min()
        )

        # comp self-attention map
        if self.args.mean_comp:
            self_attn = np.mean(vh_, axis=0)
            self.print(f"use the mean of {self.args.max_com} comps.")
        else:
            self_attn = vh_[self.args.comp_idx]
            self.print(f"select {self.args.comp_idx}-th comp.")
        # to [0, 1]
        self_attn = (self_attn - self_attn.min()) / (self_attn.max() - self_attn.min())
        # visual final self-attention
        self_attn_vis = np.copy(self_attn)
        self_attn_vis = self_attn_vis * 255
        self_attn_vis = np.repeat(
            np.expand_dims(self_attn_vis, axis=2), 3, axis=2
        ).astype(np.uint8)
        view_images(
            self_attn_vis, save_image=True, fp=self.results_path / "self-attn-final.png"
        )
        """attention map fusion"""
        attn_map = (
            self.args.attn_coeff * cross_attn_map
            + (1 - self.args.attn_coeff) * self_attn
        )
        # to [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        self.print(f"-> fusion attn_map: {attn_map.shape}")
        attention_collect_timelog.finish()

        return target_path, attn_map

    def get_target(self, target_file, image_size):
        if not is_image_file(str(target_file)):
            raise TypeError(f"{target_file} is not image file.")

        target = Image.open(target_file)

        if target.mode == "RGBA":
            # Create a white rgba background
            new_image = Image.new("RGBA", target.size, "WHITE")
            # Paste the image on the background.
            new_image.paste(target, (0, 0), target)
            target = new_image
        target = target.convert("RGB")

        # define image transforms
        transforms_ = []
        if target.size[0] != target.size[1]:
            transforms_.append(transforms.Resize((image_size, image_size)))
        else:
            transforms_.append(transforms.Resize(image_size))
            transforms_.append(transforms.CenterCrop(image_size))
        transforms_.append(transforms.ToTensor())

        # preprocess
        data_transforms = transforms.Compose(transforms_)
        target_ = data_transforms(target).unsqueeze(0).to(self.device)

        return target_
