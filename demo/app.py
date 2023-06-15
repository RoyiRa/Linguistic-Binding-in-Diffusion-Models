import gradio as gr
import torch

from syngen_diffusion_pipeline import SynGenDiffusionPipeline

import subprocess

def install_spacy_model(model_name):
    try:
        subprocess.check_call(["python", "-m", "pip", "install", "spacy"])
        subprocess.check_call(["python", "-m", "spacy", "download", model_name])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing the model: {model_name}")
        print(f"Error details: {str(e)}")

install_spacy_model("en_core_web_trf")

model_path = 'CompVis/stable-diffusion-v1-4'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
pipe = SynGenDiffusionPipeline.from_pretrained(model_path).to(device)


def generate_fn(prompt, seed):
    generator = torch.Generator(device.type).manual_seed(int(seed))
    result = pipe(prompt=prompt, generator=generator, num_inference_steps=50)
    return result['images'][0]

title = "SynGen"
description = """
This is the demo for [SynGen](https://github.com/RoyiRa/Syntax-Guided-Generation), an image synthesis approach which first syntactically analyses the prompt to identify entities and their modifiers, and then uses a novel loss function that encourages the cross-attention maps to agree with the linguistic binding reflected by the syntax. Preprint: \"Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment\" (arxiv link coming soon).
"""

examples = [
    ["a yellow flamingo and a pink sunflower", "16"],
    ["a yellow flamingo and a pink sunflower", "60"],
    ["a checkered bowl in a cluttered room", "69"],
    ["a checkered bowl in a cluttered room", "77"],
    ["a horned lion and a spotted monkey", "1269"],
    ["a horned lion and a spotted monkey", "9146"]
]

prompt_textbox = gr.Textbox(label="Prompt", placeholder="A yellow flamingo and a pink sunflower", lines=1)
seed_textbox = gr.Textbox(label="Seed", placeholder="42", lines=1)

output = gr.Image(label="generation")
demo = gr.Interface(fn=generate_fn, inputs=[prompt_textbox, seed_textbox], outputs=output, examples=examples,
                    title=title, description=description, allow_flagging=False)

demo.launch()
