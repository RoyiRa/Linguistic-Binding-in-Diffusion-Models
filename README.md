# Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment

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
