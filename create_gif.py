from PIL import Image
import os
import argparse
import cairosvg
from io import BytesIO

from pathlib import Path

def create_gif_from_svgs(input_path, output_path=None):
    print("Getting Images from",input_path)
    
    if output_path==None:
        output_path = input_path
    
    input_path = Path(input_path)
    
    svg_paths = [input_path / name for name in sorted(os.listdir(input_path))]
    
    
    images = [Image.open(BytesIO(cairosvg.svg2png(url=str(svg_path)))) for svg_path in svg_paths]

        # Save the first image with a .gif extension and append the rest
    gif_path = output_path / 'creation.gif'  # Specify the path for the output GIF
    images[0].save(gif_path, save_all=True, append_images=images, optimize=False, duration=200, loop=0)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="vectorfusion rendering"
    )
    # config
    parser.add_argument("--input_path",
                        required=True, type=str,
                        default="",
                        help="input_path")
    parser.add_argument("--output_path",
                        required=False, type=str,
                        default="",
                        help="output_path")
    args = parser.parse_args()
    create_gif_from_svgs(args.input_path,args.output_path)
    