# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import copy
import math
import random
import pathlib
from typing import List, Dict

from shapely.geometry.polygon import Polygon
from omegaconf import DictConfig
import numpy as np
import pydiffvg
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from methods.painter.vectorfusion.coord_inits import RandomCoordInit,NaiveCoordInit,SparseCoordInit, AttentionCoordInit


class Painter(nn.Module):

    def __init__(
            self,
            style: str,
            target_img: torch.Tensor,
            num_segments: int,
            segment_init: str,
            radius: int = 20,
            canvas_size: int = 600,
            n_grid: int = 32,
            trainable_bg: bool = False,
            stroke: bool = False,
            stroke_width: int = 3,
            path_svg=None,
            device=None,
            attention_map =None,
            softmax_temp = None,
            num_stages = None,
            num_paths = None,
            xdog_intersec = None

        
    ):
        super(Painter, self).__init__()
        self.device = device
        self.target_img = target_img

        self.style = style
        assert style in ["iconography", "pixelart", "sketch"]

        self.num_segments = num_segments
        self.segment_init = segment_init
        self.radius = radius

        self.canvas_width, self.canvas_height = canvas_size, canvas_size
        """pixelart params"""
        self.n_grid = n_grid  # divide the canvas into n grids
        self.pixel_per_grid = self.canvas_width // self.n_grid
        """sketch params"""
        self.stroke_width = stroke_width
        """iconography params"""
        self.color_ref = None
        self.train_stroke = stroke

        self.shapes = []  # record all paths
        self.shape_groups = []
        self.cur_shapes, self.cur_shape_groups = [], []  # record the current optimized path
        self.points_vars = []
        self.color_vars = []
        self.stroke_width_vars = []
        self.stroke_color_vars = []

        self.path_svg = path_svg
        self.optimize_flag = []

        self.strokes_counter = 0  # counts the number of calls to "get_path"

        # Background color
        self.para_bg = torch.tensor([1., 1., 1.], requires_grad=trainable_bg, device=self.device)

        #attention init
        self.pos_init_method = None
        self.attention_map = attention_map
        self.softmax_temp = softmax_temp
        self.num_stages = num_stages
        self.num_paths = num_paths
        self.xdog_intersec = xdog_intersec

    def component_wise_path_init(self, pred, init_type: str = 'sparse'):
        assert self.target_img is not None  # gt

        if init_type == 'random':
            self.pos_init_method = RandomCoordInit(self.canvas_height, self.canvas_width)
        elif init_type == 'sparse':
            # when initialized for the first time, the render result is None
            if pred is None:
                pred = self.para_bg.view(1, -1, 1, 1).repeat(1, 1, self.canvas_height, self.canvas_width)
            # then pred is the render result
            self.pos_init_method = SparseCoordInit(pred, self.target_img)
        elif init_type == 'naive':
            if pred is None:
                pred = self.para_bg.view(1, -1, 1, 1).repeat(1, 1, self.canvas_height, self.canvas_width)
            self.pos_init_method = NaiveCoordInit(pred, self.target_img)
        elif init_type == 'attention':
            self.pos_init_method = AttentionCoordInit(attention_map=self.attention_map, 
                                                      softmax_temp=self.softmax_temp,
                                                      canvas_height=self.canvas_height,
                                                      canvas_width= self.canvas_width, 
                                                      num_paths=self.num_paths, 
                                                      num_stages=self.num_stages,
                                                      image2clip_input = self.target_img,
                                                      xdog_intersec = self.xdog_intersec)
            
        elif init_type == 'clipPasso':
            raise NotImplementedError(f"'{init_type}' is not implemented.")
        else:
            raise NotImplementedError(f"'{init_type}' is not support.")

    def init_image(self, stage=0, num_paths=0):
        self.cur_shapes, self.cur_shape_groups = [], []

        if self.style == 'pixelart':  # update path definition
            num_paths = self.n_grid

        if stage > 0:
            # Noting: if multi stages training than add new strokes on existing ones
            # don't optimize on previous strokes
            self.optimize_flag = [False for i in range(len(self.shapes))]
            for i in range(num_paths):
                if self.style == 'iconography':
                    path = self.get_path()
                    self.shapes.append(path)
                    self.cur_shapes.append(path)

                    fill_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                    fill_color_init[-1] = np.random.uniform(0.7, 1)
                    stroke_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([self.strokes_counter - 1]),
                        fill_color=fill_color_init,
                        stroke_color=stroke_color_init
                    )
                    self.shape_groups.append(path_group)
                    self.cur_shape_groups.append(path_group)
                    self.optimize_flag.append(True)

                elif self.style == 'pixelart':
                    for j in range(num_paths):
                        path = self.get_path(coord=[i, j])
                        self.shapes.append(path)
                        self.cur_shapes.append(path)

                        fill_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                        fill_color_init[-1] = np.random.uniform(0.7, 1)
                        stroke_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                        path_group = pydiffvg.ShapeGroup(
                            shape_ids=torch.LongTensor([i * num_paths + j]),
                            fill_color=fill_color_init,
                            stroke_color=stroke_color_init,
                        )
                        self.shape_groups.append(path_group)
                        self.cur_shape_groups.append(path_group)
                        self.optimize_flag.append(True)

                elif self.style == 'sketch':
                    path = self.get_path()
                    self.shapes.append(path)
                    self.cur_shapes.append(path)

                    stroke_color_init = torch.tensor([0.0, 0.0, 0.0, 1.0])
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([self.strokes_counter - 1]),
                        fill_color=None,
                        stroke_color=stroke_color_init
                    )
                    self.shape_groups.append(path_group)
                    self.cur_shape_groups.append(path_group)
                    self.optimize_flag.append(True)
        else:
            num_paths_exists = 0
            if self.path_svg is not None and pathlib.Path(self.path_svg).exists():
                print(f"-> init svg from `{self.path_svg}` ...")

                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = self.load_svg(self.path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)

                self.cur_shapes = self.shapes
                self.cur_shape_groups = self.shape_groups

            for i in range(num_paths_exists, num_paths):
                if self.style == 'iconography':
                    path = self.get_path()
                    self.shapes.append(path)
                    self.cur_shapes.append(path)

                    wref, href = self.color_ref
                    wref = max(0, min(int(wref), self.canvas_width - 1))
                    href = max(0, min(int(href), self.canvas_height - 1))
                    fill_color_init = list(self.target_img[0, :, href, wref]) + [1.]
                    fill_color_init = torch.FloatTensor(fill_color_init)
                    stroke_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([self.strokes_counter - 1]),
                        fill_color=None if self.train_stroke else fill_color_init,
                        stroke_color=stroke_color_init if self.train_stroke else None
                    )
                    self.shape_groups.append(path_group)
                    self.cur_shape_groups.append(path_group)

                elif self.style == 'pixelart':
                    for j in range(num_paths):
                        path = self.get_path(coord=[i, j])
                        self.shapes.append(path)
                        self.cur_shapes.append(path)

                        fill_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                        fill_color_init[-1] = np.random.uniform(0.7, 1)
                        stroke_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                        path_group = pydiffvg.ShapeGroup(
                            shape_ids=torch.LongTensor([i * num_paths + j]),
                            fill_color=fill_color_init,
                            stroke_color=stroke_color_init,
                        )
                        self.shape_groups.append(path_group)
                        self.cur_shape_groups.append(path_group)

                elif self.style == 'sketch':
                    path = self.get_path()
                    self.shapes.append(path)
                    self.cur_shapes.append(path)

                    stroke_color_init = torch.tensor([0.0, 0.0, 0.0, 1.0])
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([len(self.shapes) - 1]),
                        fill_color=None,
                        stroke_color=stroke_color_init
                    )
                    self.shape_groups.append(path_group)
                    self.cur_shape_groups.append(path_group)

            self.optimize_flag = [True for i in range(len(self.shapes))]

        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + self.para_bg * (1 - img[:, :, 3:4])
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_image(self, step: int = 0):
        img = self.render_warp(step)
        img = img[:, :, 3:4] * img[:, :, :3] + self.para_bg * (1 - img[:, :, 3:4])
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_path(self, coord=None):
        num_segments = self.num_segments

        points = []
        if self.style == 'iconography':
            # init segment
            if self.segment_init == 'circle':
                num_control_points = [2] * num_segments
                radius = self.radius if self.radius is not None else np.random.uniform(0.5, 1)
                if self.pos_init_method is not None:
                    center = self.pos_init_method()
                else:
                    center = (random.random(), random.random())
                bias = center
                self.color_ref = copy.deepcopy(bias)

                avg_degree = 360 / (num_segments * 3)
                for i in range(0, num_segments * 3):
                    point = (
                        np.cos(np.deg2rad(i * avg_degree)), np.sin(np.deg2rad(i * avg_degree))
                    )
                    points.append(point)

                points = torch.FloatTensor(points) * radius + torch.FloatTensor(bias).unsqueeze(dim=0)
            elif self.segment_init == 'random':
                num_control_points = [2] * num_segments
                p0 = self.pos_init_method()
                self.color_ref = copy.deepcopy(p0)
                points.append(p0)

                for j in range(num_segments):
                    radius = self.radius
                    p1 = (p0[0] + radius * np.random.uniform(-0.5, 0.5),
                          p0[1] + radius * np.random.uniform(-0.5, 0.5))
                    p2 = (p1[0] + radius * np.random.uniform(-0.5, 0.5),
                          p1[1] + radius * np.random.uniform(-0.5, 0.5))
                    p3 = (p2[0] + radius * np.random.uniform(-0.5, 0.5),
                          p2[1] + radius * np.random.uniform(-0.5, 0.5))
                    points.append(p1)
                    points.append(p2)
                    if j < num_segments - 1:
                        points.append(p3)
                        p0 = p3
                points = torch.FloatTensor(points)
            else:
                raise NotImplementedError(f"{self.segment_init} is not exists.")

            path = pydiffvg.Path(
                num_control_points=torch.LongTensor(num_control_points),
                points=points,
                stroke_width=torch.tensor(float(self.stroke_width)) if self.train_stroke else torch.tensor(0.0),
                is_closed=True
            )
        elif self.style == 'sketch':
            num_control_points = torch.zeros(num_segments, dtype=torch.long) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)

            for j in range(num_segments):
                radius = 0.1
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points).to(self.device)
            points[:, 0] *= self.canvas_width
            points[:, 1] *= self.canvas_height

            path = pydiffvg.Path(num_control_points=torch.LongTensor(num_control_points),
                                 points=points,
                                 stroke_width=torch.tensor(float(self.stroke_width)),
                                 is_closed=False)
        elif self.style == 'pixelart':
            x = coord[0] * self.pixel_per_grid
            y = coord[1] * self.pixel_per_grid
            points = torch.FloatTensor([
                [x, y],
                [x + self.pixel_per_grid, y],
                [x + self.pixel_per_grid, y + self.pixel_per_grid],
                [x, y + self.pixel_per_grid]
            ]).to(self.device)
            path = pydiffvg.Polygon(points=points,
                                    stroke_width=torch.tensor(0.0),
                                    is_closed=True)

        self.strokes_counter += 1
        return path

    def clip_curve_shape(self):
        for group in self.shape_groups:
            if self.train_stroke:
                group.stroke_color.data.clamp_(0.0, 1.0)
            else:
                group.fill_color.data.clamp_(0.0, 1.0)

    def reinitialize_paths(self,
                           reinit_path: bool = False,
                           opacity_threshold: float = None,
                           area_threshold: float = None,
                           fpath: pathlib.Path = None):
        """
        reinitialize paths, also known as 'Reinitializing paths' in VectorFusion paper.

        Args:
            reinit_path: whether to reinitialize paths or not.
            opacity_threshold: Threshold of opacity.
            area_threshold: Threshold of the closed polygon area.
            fpath: The path to save the reinitialized SVG.
        """
        if self.style == 'iconography' and reinit_path and (not self.train_stroke):
            # re-init by opacity_threshold
            select_path_ids_by_opc = []
            if opacity_threshold != 0 and opacity_threshold is not None:
                def get_keys_below_threshold(my_dict, threshold):
                    keys_below_threshold = [key for key, value in my_dict.items() if value < threshold]
                    return keys_below_threshold

                opacity_record_ = {group.shape_ids.item(): group.fill_color.data[-1].item()
                                   for group in self.cur_shape_groups}
                # print("-> opacity_record: ", opacity_record_)
                print("-> opacity_record: ", [f"{k}: {v:.3f}" for k, v in opacity_record_.items()])
                select_path_ids_by_opc = get_keys_below_threshold(opacity_record_, opacity_threshold)
                print("select_path_ids_by_opc: ", select_path_ids_by_opc)

            # remove path by area_threshold
            select_path_ids_by_area = []
            if area_threshold != 0 and area_threshold is not None:
                area_records = [Polygon(shape.points.detach().numpy()).area for shape in self.cur_shapes]
                # print("-> area_records: ", area_records)
                print("-> area_records: ", ['%.2f' % i for i in area_records])
                for i, shape in enumerate(self.cur_shapes):
                    if Polygon(shape.points.detach().numpy()).area < area_threshold:
                        select_path_ids_by_area.append(shape.id)
                print("select_path_ids_by_area: ", select_path_ids_by_area)

            # re-init paths
            reinit_union = list(set(select_path_ids_by_opc + select_path_ids_by_area))
            if len(reinit_union) > 0:
                for i, path in enumerate(self.cur_shapes):
                    if path.id in reinit_union:
                        self.cur_shapes[i] = self.get_path()
                for i, group in enumerate(self.cur_shape_groups):
                    shp_ids = group.shape_ids.cpu().numpy().tolist()
                    if set(shp_ids).issubset(reinit_union):
                        fill_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                        fill_color_init[-1] = np.random.uniform(0.7, 1)
                        stroke_color_init = torch.FloatTensor(np.random.uniform(size=[4]))
                        self.cur_shape_groups[i] = pydiffvg.ShapeGroup(
                            shape_ids=torch.tensor(list(shp_ids)),
                            fill_color=fill_color_init,
                            stroke_color=stroke_color_init)
                # save reinit svg
                self.save_svg(fpath)

            print("-" * 40)

    def render_warp(self, seed=0):
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups
        )
        _render = pydiffvg.RenderFunction.apply
        img = _render(self.canvas_width,  # width
                      self.canvas_height,  # height
                      2,  # num_samples_x
                      2,  # num_samples_y
                      seed,  # seed
                      None,
                      *scene_args)
        return img

    def calc_distance_weight(self, loss_weight_keep):
        shapes_forsdf = copy.deepcopy(self.cur_shapes)
        shape_groups_forsdf = copy.deepcopy(self.cur_shape_groups)
        for si in shapes_forsdf:
            si.stroke_width = torch.FloatTensor([0]).to(self.device)
        for sg_idx, sgi in enumerate(shape_groups_forsdf):
            sgi.fill_color = torch.FloatTensor([1, 1, 1, 1]).to(self.device)
            sgi.shape_ids = torch.LongTensor([sg_idx]).to(self.device)

        sargs_forsdf = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, shapes_forsdf, shape_groups_forsdf
        )
        _render = pydiffvg.RenderFunction.apply
        with torch.no_grad():
            im_forsdf = _render(self.canvas_width,  # width
                                self.canvas_height,  # height
                                2,  # num_samples_x
                                2,  # num_samples_y
                                0,  # seed
                                None,
                                *sargs_forsdf)

        # use alpha channel is a trick to get 0-1 image
        im_forsdf = (im_forsdf[:, :, 3]).detach().cpu().numpy()
        loss_weight = get_sdf(im_forsdf, normalize='to1')
        loss_weight += loss_weight_keep
        loss_weight = np.clip(loss_weight, 0, 1)
        loss_weight = torch.FloatTensor(loss_weight).to(self.device)
        return loss_weight

    def set_points_parameters(self, id_delta=0):
        # stroke`s location optimization
        self.points_vars = []
        self.stroke_width_vars = []
        for i, path in enumerate(self.cur_shapes):
            path.id = i + id_delta  # set point id
            path.points.requires_grad = True
            self.points_vars.append(path.points)

            if self.train_stroke:
                path.stroke_width.requires_grad = True
                self.stroke_width_vars.append(path.stroke_width)

    def set_color_parameters(self):
        # for stroke' color optimization
        self.color_vars = []
        self.stroke_color_vars = []
        for i, group in enumerate(self.cur_shape_groups):
            group.fill_color.requires_grad = True
            self.color_vars.append(group.fill_color)

            if self.train_stroke:
                group.stroke_color.requires_grad = True
                self.stroke_color_vars.append(group.stroke_color)

    def get_point_parameters(self):
        return self.points_vars

    def get_color_parameters(self):
        return self.color_vars

    def get_stroke_parameters(self):
        return self.stroke_width_vars, self.stroke_color_vars

    def get_bg_parameters(self):
        return self.para_bg

    def save_svg(self, fpath):
        pydiffvg.save_svg(f'{fpath}',
                          self.canvas_width,
                          self.canvas_height,
                          self.shapes,
                          self.shape_groups)

    def load_svg(self, path_svg):
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(path_svg)
        return canvas_width, canvas_height, shapes, shape_groups


def get_sdf(phi, **kwargs):
    import skfmm  # local import

    phi = (phi - 0.5) * 2
    if (phi.max() <= 0) or (phi.min() >= 0):
        return np.zeros(phi.shape).astype(np.float32)
    sd = skfmm.distance(phi, dx=1)

    flip_negative = kwargs.get('flip_negative', True)
    if flip_negative:
        sd = np.abs(sd)

    truncate = kwargs.get('truncate', 10)
    sd = np.clip(sd, -truncate, truncate)
    # print(f"max sd value is: {sd.max()}")

    zero2max = kwargs.get('zero2max', True)
    if zero2max and flip_negative:
        sd = sd.max() - sd
    elif zero2max:
        raise ValueError

    normalize = kwargs.get('normalize', 'sum')
    if normalize == 'sum':
        sd /= sd.sum()
    elif normalize == 'to1':
        sd /= sd.max()
    return sd

class PainterOptimizer:

    def __init__(self,
                 renderer: Painter,
                 style: str,
                 num_iter: int,
                 lr_config: DictConfig,
                 trainable_stroke: bool = False,
                 trainable_bg: bool = False):
        self.renderer = renderer
        self.num_iter = num_iter
        self.trainable_stroke = trainable_stroke
        self.trainable_bg = trainable_bg

        self.lr_base = {
            'point': lr_config.point,
            'color': lr_config.fill_color,
            'stroke_width': lr_config.stroke_width,
            'stroke_color': lr_config.stroke_color,
            'bg': lr_config.bg
        }

        self.learnable_params = []  # list[Dict]

        self.style = style
        if style == 'iconography':
            if self.trainable_stroke:
                self.optim_point, self.optim_color = True, False
            else:
                self.optim_point, self.optim_color = True, True
            self.lr_lambda = LinearDecayLRLambda(self.num_iter, decay_ratio=0.4)
        if style == 'pixelart':
            self.optim_point, self.optim_color = False, True
            self.lr_lambda = LinearDecayLRLambda(self.num_iter, decay_ratio=0.4)
        if style == 'sketch':
            self.optim_point, self.optim_color = True, False
            self.lr_lambda = SketchLRLambda(self.num_iter,
                                            warmup_steps=500, warmup_start_lr=0.02, warmup_end_lr=0.2,
                                            cosine_end_lr=0.05)

        self.optimizer = None
        self.scheduler = None

    def init_optimizers(self, pid_delta: int = 0, add_params: List[Dict] = None):
        # optimizer
        params = {}
        if self.optim_point:
            self.renderer.set_points_parameters(pid_delta)
            params['point'] = self.renderer.get_point_parameters()

        if self.optim_color:
            self.renderer.set_color_parameters()
            params['color'] = self.renderer.get_color_parameters()

        if self.trainable_bg:
            params['bg'] = self.renderer.get_bg_parameters()

        if self.trainable_stroke:
            params['stroke_width'], params['stroke_color'] = self.renderer.get_stroke_parameters()

        self.learnable_params = [
            {'params': params[ki], 'lr': self.lr_base[ki], '_id': str(ki)} for ki in sorted(params.keys())
        ]
        if add_params is not None:
            self.learnable_params.extend(add_params)

        self.optimizer = torch.optim.Adam(self.learnable_params, betas=(0.9, 0.9), eps=1e-6)

        # lr schedule
        if self.lr_lambda is not None:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

    def update_params(self, name: str, value: torch.tensor):
        for param_group in self.learnable_params:
            if param_group.get('_id') == name:
                param_group['params'] = value

    def update_lr(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad_(self):
        self.optimizer.zero_grad()

    def step_(self):
        self.optimizer.step()

    def get_lr(self) -> Dict:
        lr = {}
        for _group in self.optimizer.param_groups:
            lr[_group['_id']] = _group['lr']
        return lr


class LinearDecayLRLambda:

    def __init__(self, decay_every, decay_ratio):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n):
        decay_time = n // self.decay_every
        decay_step = n % self.decay_every
        lr_s = self.decay_ratio ** decay_time
        lr_e = self.decay_ratio ** (decay_time + 1)
        r = decay_step / self.decay_every
        lr = lr_s * (1 - r) + lr_e * r
        return lr


class SketchLRLambda:
    def __init__(self, num_steps, warmup_steps, warmup_start_lr, warmup_end_lr, cosine_end_lr):
        self.n_steps = num_steps
        self.n_warmup = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.cosine_end_lr = cosine_end_lr

    def __call__(self, n):
        if n < self.n_warmup:
            # linearly warmup
            return self.warmup_start_lr + (n / self.n_warmup) * (self.warmup_end_lr - self.warmup_start_lr)
        else:
            # cosine decayed schedule
            return self.cosine_end_lr + 0.5 * (self.warmup_end_lr - self.cosine_end_lr) * (
                    1 + math.cos(math.pi * (n - self.n_warmup) / (self.n_steps - self.n_warmup)))


class PixelArtLRLambdaF:
    def __init__(self, num_steps, lr_base):
        self.num_steps = num_steps
        self.lr_base = lr_base
        self.max_lr = lr_base * 10

    def __call__(self, n):
        return self.lr_base + (self.max_lr - self.lr_base) * n / self.num_steps
