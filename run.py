import argparse
import os

import torch
from syngen_diffusion_pipeline import SynGenDiffusionPipeline


def main(prompt, seed, output_directory, model_path):
    pipe = load_model(model_path)
    image = generate(pipe, prompt, seed)
    save_image(image, prompt, seed, output_directory)


def load_model(model_path):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pipe = SynGenDiffusionPipeline.from_pretrained(model_path).to(device)

    return pipe


def generate(pipe, prompt, seed):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator(device.type).manual_seed(seed)
    result = pipe(prompt=prompt, generator=generator)
    return result['images'][0]


def save_image(image, prompt, seed, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_name = f"{output_directory}/{prompt}_{seed}.png"
    image.save(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="a checkered bowl on a red and blue table"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1924
    )

    parser.add_argument(
        '--output_directory',
        type=str,
        default='./output'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='CompVis/stable-diffusion-v1-4',
        help='The path to the model (this will download the model if the path doesn\'t exist)'
    )

    args = parser.parse_args()

    main(args.prompt, args.seed, args.output_directory, args.model_path)
