import os
import PIL
import argparse
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
import spacy


nlp = spacy.load('en_core_web_trf')


def custom_extract_attribution_indices(prompt, parser):
    doc = parser(prompt)
    subtrees = []
    modifiers = ['amod', 'nmod', 'compound', 'npadvmod', 'det']

    for w in doc:
        if w.pos_ not in ['NOUN', 'PROPN'] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == 'conj':
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees


def custom_segment_text(text: str):
    segments = []
    doc = nlp(text)
    subtrees = custom_extract_attribution_indices(doc, nlp)
    if subtrees:
        for subtree in subtrees:
            segments.append(" ".join([t.text for t in subtree]))
    return segments


def segment_text_for_automatic_eval(text: str):
    segments = custom_segment_text(text)
    doc = nlp(text)

    # add empty trees if needed
    for w in doc:
        if w.pos_ in ['NOUN', 'PROPN']:
            found_in_segment = False
            for segment in segments:
                if w.text in segment:
                    found_in_segment = True
                    break
            if not found_in_segment:
                segments.append(w.text)
    print(f"Segments: {segments}")
    return segments


class CLIPSimilarity:
    def __init__(self, clip_type='openai/clip-vit-base-patch32', device=None):
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_type = clip_type

    def get_similarity_score(self, text, image):
        text = segment_text_for_automatic_eval(text)
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True)
        outputs = self.clip(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=0)  # we can take the softmax to get the label probabilities
        scores = torch.mean(probs, dim=1)
        return scores


def get_scorable_images(images_dir, relevant_captions):
    scorable_images = []

    images_dirs = [os.path.join(images_dir, caption) for caption in os.listdir(images_dir) if
                   os.path.isdir(os.path.join(images_dir, caption))]

    print(f"Found {len(images_dirs)} images dirs.")
    for caption_dir in images_dirs:
        caption = caption_dir.split("/")[-1].replace("'", "_").replace("_", " ")
        if caption not in relevant_captions:
            continue

        images = []
        models = []
        seed = None
        for image_name in os.listdir(caption_dir):
            model = image_name.split("_")[0]
            if not seed:
                if '.jpg' in image_name:
                    seed = int(image_name.split("_")[1][:-4])
                else:
                    seed = int(image_name.split("_")[-1])
            image_path = os.path.join(images_dir, caption_dir, image_name)
            # image = PIL.Image.open(image_path)
            with PIL.Image.open(image_path) as image:
                images.append(image.copy())

            # images.append(image)
            models.append(model)



        assert len(models) == len(images)

        scorable_images.append({
            'caption': caption,
            'seed': seed,
            "images": images,
            "models": models,
        })
    cap = [s['caption'] for s in scorable_images]
    for caption in relevant_captions:
        if caption not in cap:
            print(caption)
    return scorable_images


def load_evaluatable_images(majority_path, images_dir, exclude_no_clear_winner=True):
    majority_choices = pd.read_csv(majority_path)
    if 'caption_type' in majority_choices.columns:
        majority_choices = majority_choices[majority_choices['caption_type'] != 'animal_animal']

    if exclude_no_clear_winner and 'human_annotation' in majority_choices.columns:
        concept_mask = ~majority_choices['human_annotation'].isin(['equally good', 'equally bad', 'Undecided'])
        filtered_majority_choices = majority_choices[concept_mask]
        decided_captions = [caption.replace("'", " ") for caption in filtered_majority_choices['caption'].tolist()]
    else:
        decided_captions = [caption.replace("'", " ") for caption in majority_choices['caption'].tolist()]

    scorable_images = get_scorable_images(images_dir, decided_captions)
    print(len(scorable_images), len(decided_captions))
    assert len(decided_captions) == len(scorable_images)

    # Update the list of dictionaries with the human_annotation value from the dataframe
    if 'human_annotation' in majority_choices.columns:
        # Create a dictionary from the dataframe where the caption column is the index
        concept_maj_dict = majority_choices.set_index('caption')['human_annotation'].to_dict()
        for scorable in scorable_images:
            scorable['human_annotation'] = concept_maj_dict.get(scorable['caption'], None)
    return scorable_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Evaluation Parameters')
    parser.add_argument('--captions_and_labels', type=str, required=True,
                        help='Path to the CSV file containing captions and labels.')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to the directory containing image subdirectories.')

    args = parser.parse_args()
    captions_and_labels = args.captions_and_labels
    images_dir = args.images_dir


    data = load_evaluatable_images(captions_and_labels, images_dir, exclude_no_clear_winner=False)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    scorer = CLIPSimilarity(device=device)
    score_counter = {data[0]['models'][i]: 0 for i in range(len(data[0]['models']))}
    for idx, image in enumerate(data):
        image['scores'] = scorer.get_similarity_score(image['caption'], image['images'])
        image['scores'] = [score.item() for score in image['scores']]
        scores = image['scores']
        max_index = scores.index(max(scores))
        score_counter[image['models'][max_index]] += 1
        data[idx] = image
    print(score_counter)