# -*- coding: utf-8 -*-
# Author: ximing
# Description: the main func of this project.
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

#python evaluate.py -c vectorfusion.yaml --generate_eval_data --test_data_name=test1 -update="${PARAMS}" -d 8888 --download -respath /tmp/workdir

import os
import sys
import argparse
import ast
import wandb
from PIL import Image
from pathlib import Path
import cairosvg

from accelerate.utils import set_seed
from methods.painter.vectorfusion.utils import TimeLogger
import omegaconf
import csv
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from libs.engine import merge_and_update_config
from libs.utils.argparse import accelerate_parser, base_data_parser

import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Set device to CUDA
    device = torch.device("cuda")
    print("Using CUDA (GPU)")
else:
    # Set device to CPU
    device = torch.device("cpu")
    print("Using CPU")

def get_prompts():
    prompts = []
    with open('/testdata/data.csv',encoding="UTF-8") as file_obj:

        # Create reader object by passing the file  
        # object to reader method 
        reader_obj = csv.reader(file_obj,delimiter=";") 

        # Iterate over each row in the csv  
        # file using reader object 
        for idx, row in enumerate(reader_obj): 
            if idx == 0:
                continue
            prompts.append(row[0])
            
    return prompts

def create_output_folder(dataset_path):
    
    if not os.path.exists(dataset_path):
        # Create the folder if it does not exist
        os.makedirs(dataset_path)
        os.makedirs(dataset_path / "svgs")
        os.makedirs(dataset_path / "pngs")
    else:
        # Raise an exception if the folder already exists
        raise FileExistsError(f"The folder {dataset_path} already exists. Dont overwrite our stuff. >:(")
    

def change_args_for_generation(args):

    args.use_wandb = False
    args.negative_prompt = None
    args.save_step = args.sds.num_iter
    return args

def evaluate(args):

    from pipelines.painter.VectorFusion_pipeline import VectorFusionPipeline
    
    from pipelines.painter.VectorFusion_pipeline import get_clip_score
    
    args.batch_size = 1  # rendering one SVG at a time
    
    #init wandb 
    if args.use_wandb:
        wandb.init(entity="aiis-chair")
    
    total_log= TimeLogger(name="total",use_wandb=args.use_wandb)
    
    print("args.test_data_name",args.test_data_name)

    if args.path_schedule == 'list' and not isinstance(args.schedule_each, omegaconf.ListConfig):
        args.schedule_each = ast.literal_eval(args.schedule_each)
    if not isinstance(args.sds.t_range, omegaconf.ListConfig):
        args.sds.t_range = ast.literal_eval(args.sds.t_range)
        

    
    
    #load data 
    prompts = get_prompts()
    
    dataset_path = Path("/testdata") / args.test_data_name
    
    if args.generate_eval_data:
        
        create_output_folder(dataset_path)
        
        for idx, prompt in enumerate(prompts,start=1): 
            
            args = change_args_for_generation(args)
            init_log= TimeLogger(name="init",use_wandb=args.use_wandb)
            pipe = VectorFusionPipeline(args,eval_mode=True)
            init_log.finish()
            if idx==1:
                args.download=False
            paint_log = TimeLogger(name="paint",use_wandb=args.use_wandb)
            svg_output_path = dataset_path / "svgs"
            pipe.paint_for_evaluation(prompt,svg_output_path,str(idx))
            paint_log.finish()

    
    #iterate over dataset 
    total_score = 0
    for idx, prompt in enumerate(prompts,start=1):
        
        print(prompt)
        png_path = dataset_path / f"pngs/{idx}.png" 
        if args.generate_eval_data:
            svg_path = dataset_path / f"svgs/{idx}.svg" 
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
        clip_score = get_clip_score(prompt,Image.open(png_path),device)
        if args.use_wandb:
            wandb.log({"idx":idx,"clip_socre":clip_score})
        print(clip_score)
        total_score+=clip_score
    total_score /= len(prompts)
    
    print("total_score",total_score)
    
    total_log.finish()
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="vectorfusion rendering",
        parents=[accelerate_parser(), base_data_parser()]
    )
    # config
    parser.add_argument("-c", "--config",
                        required=True, type=str,
                        default="",
                        help="YAML/YML file for configuration.")
    # DiffSVG
    parser.add_argument("--print_timing", "-timing", action="store_true",
                        help="set print svg rendering timing.")
    # diffuser
    parser.add_argument("--download", action="store_true",
                        help="download models from huggingface automatically.")
    parser.add_argument("--force_download", "-download", action="store_true",
                        help="force the models to be downloaded from huggingface.")
    parser.add_argument("--resume_download", "-dpm_resume", action="store_true",
                        help="download the models again from the breakpoint.")
    #evaluation 
    parser.add_argument("--test_data_name",type=str )
    parser.add_argument("--generate_eval_data",  action="store_true")

    args = parser.parse_args()
    args = merge_and_update_config(args)

    set_seed(args.seed)
    evaluate(args)