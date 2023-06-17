# Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment

This repository hosts the code and resources associated with our [paper](https://arxiv.org/abs/2306.08877) on linguistic binding in diffusion models.

## Abstract
Text-conditioned image generation models often generate incorrect associations between entities and their visual attributes. This reflects an impaired mapping between linguistic binding of entities and modifiers in the prompt and visual binding of the corresponding elements in the generated image. As one notable example, a query like ``a pink sunflower and a yellow flamingo'' may incorrectly produce an image of a yellow sunflower and a pink flamingo. To remedy this issue, we propose SynGen, an approach which first syntactically analyses the prompt to identify entities and their modifiers, and then uses a novel loss function that encourages the cross-attention maps to agree with the linguistic binding reflected by the syntax. Specifically, we encourage large overlap between attention maps of entities and their modifiers, and small overlap with other entities and modifier words. The loss is optimized during inference, without retraining or fine-tuning the model. Human evaluation on three datasets, including one new and challenging set, demonstrate significant improvements of SynGen compared with current state of the art methods. This work highlights how making use of sentence structure during inference can efficiently and substantially improve the faithfulness of text-to-image generation.

## Setup
Clone this repository and create a conda environment:
```
conda env create -f environment.yaml
conda activate syngen
```

If you rather use an existing environment, just run:
```
pip install -r requirements.txt
```

Finally, run:
```
python -m spacy download en_core_web_trf
```

## Inference
```
python run.py --prompt "a horned lion and a spotted monkey" --seed 1269
```

Note that this will download the stable diffusion model `CompVis/stable-diffusion-v1-4`. If you rather use an existing copy of the model, provide the absolute path using `--model_path`.

## Live Demo

Check out our [demo](https://huggingface.co/spaces/Royir/SynGen)

## Citation

If you use this code or our results in your research, please cite as:

```bibtex
@misc{rassin2023linguistic,
      title={Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment}, 
      author={Royi Rassin and Eran Hirsch and Daniel Glickman and Shauli Ravfogel and Yoav Goldberg and Gal Chechik},
      year={2023},
      eprint={2306.08877},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}


