# Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment (NeurIPS 2023, oral)

This repository hosts the code and resources associated with our [paper](https://arxiv.org/abs/2306.08877) on linguistic binding in diffusion models.

## Live Demo

Check out our [demo](https://huggingface.co/spaces/Royir/SynGen)

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

## DVMP Prompt Generation
```
python dvmp.py --num_samples 500 --dest_path destination.csv
```

### Requirements for Inputs
**num_samples**: Number of prompts to generate. Default: 200.

**dest_path**: Destination CSV file path. Default: destination.csv.


## Automatic Evaluation
```
python automatic_evaluation.py --captions_and_labels <path/to/csv/file> --images_dir <path/to/image/directory>
```

### Requirements for Inputs
**captions_and_labels**: This should be a CSV file with columns named 'caption' and 'human_annotation' (optional).

**images_dir**: This directory should have subdirectories, each named after a specific prompt given to the text-to-image model. Within each subdirectory, you should have the generated images from all the models being evaluated, following the naming convention **'{model_name}_{seed}.jpg'**.

## Citation

If you use this code or our results in your research, please cite as:

```bibtex
@article{rassin2024linguistic,
  title={Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment},
  author={Rassin, Royi and Hirsch, Eran and Glickman, Daniel and Ravfogel, Shauli and Goldberg, Yoav and Chechik, Gal},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
