import argparse
import os
import math
import torch
from syngen_diffusion_pipeline import SynGenDiffusionPipeline


def main(prompt, seed, output_directory, model_path, step_size, attn_res, include_entities):
    pipe = load_model(model_path, include_entities)
    image = generate(pipe, prompt, seed, step_size, attn_res)
    save_image(image, prompt, seed, output_directory)


def load_model(model_path, include_entities):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pipe = SynGenDiffusionPipeline.from_pretrained(model_path, include_entities=include_entities).to(device)

    return pipe


def generate(pipe, prompt, seed, step_size, attn_res):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator(device.type).manual_seed(seed)
    result = pipe(prompt=prompt, generator=generator, syngen_step_size=step_size,
                  attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))))
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

    parser.add_argument(
        '--step_size',
        type=float,
        default=20.0,
        help='The SynGen step size'
    )

    parser.add_argument(
        '--attn_res',
        type=int,
        default=256,
        help='The attention resolution (use 256 for SD 1.4, 576 for SD 2.1)'
    )

    parser.add_argument(
        '--include_entities',
        type=bool,
        default=False,
        help='Apply negative-only loss for entities with no modifiers'
    )

    args = parser.parse_args()
    main(args.prompt, args.seed, args.output_directory, args.model_path, args.step_size, args.attn_res,
         args.include_entities)
