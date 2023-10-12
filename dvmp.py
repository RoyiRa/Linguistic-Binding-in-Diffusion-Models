import random
import pandas as pd
import spacy
import argparse

nlp = spacy.load('en_core_web_trf')

def get_article(word):
    if word[0] in "aeiou":
        return "an"
    else:
        return "a"


def generate_phrase(items, modifiers, colors, item_type):
    # randomly choose num modifiers 0,1,2
    if item_type == 'fruit':
        num_modifiers = random.choice([0,1])
    else:
        num_modifiers = random.choice([0, 1, 2])
    # randomly choose num_modifiers animal_modifiers
    sampled_modifiers = random.sample(modifiers, num_modifiers)
    # randomly choose if to add color (30% chance)
    add_color = random.choice([True, False, False])
    if add_color:
        color = random.choice(colors)
        sampled_modifiers = [color] + sampled_modifiers
    # if no modifiers, try again.
    if len(sampled_modifiers) == 0:
        return generate_phrase(items, modifiers, colors, item_type)

    article = get_article(sampled_modifiers[0])
    final_modifiers = " ".join(sampled_modifiers)
    item = random.choice(items)
    return f"{article} {final_modifiers} {item}", len(sampled_modifiers)


def generate_prompt(num_phrases):
    objects = [
        "backpack", "crown", "suitcase", "chair", "balloon", "bow",
        "car", "bowl", "bench", "clock", "camera", "umbrella", "guitar", "shoe", "hat",
        "surfboard", "skateboard", "bicycle"

    ]
    fruit = ["apple", "tomato", "banana", "strawberry"]

    animals = [
        "cat", "dog", "bird", "bear", "lion", "horse", "elephant", "monkey", "frog",
        "turtle", "rabbit", "mouse", "panda", "zebra", "gorilla", "penguin"
    ]

    colors = [
        "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "gray", "black",
        "white", "beige", "teal"]

    animal_modifiers = ["furry", "baby", "spotted", "sleepy"]
    object_modifiers = ["modern", "spotted", "wooden", "metal", "curved", "spiky", "checkered"]
    fruit_modifiers = ["sliced", "skewered"]


    phrases = []
    num_modifiers = 0
    while len(phrases) < num_phrases:
        # randomly choose between animal, object, and fruit
        choice = random.choice([0, 1, 2])
        if choice == 0:
            phrase, num_phrase_modifiers = generate_phrase(animals, animal_modifiers, colors, 'animal')
        elif choice == 1:
            phrase, num_phrase_modifiers = generate_phrase(fruit, fruit_modifiers, colors, 'fruit')
        else:
            phrase, num_phrase_modifiers = generate_phrase(objects, object_modifiers, colors, 'object')

        if phrase not in phrases:
            num_modifiers += num_phrase_modifiers
            phrases.append(phrase)
    prompt = " and ".join(phrases)
    return prompt, num_modifiers



def extract_attribution_indices(prompt, parser):
    doc = parser(prompt)
    subtrees = []
    modifiers = ['amod', 'nmod', 'compound', 'npadvmod', 'advmod', 'acomp']

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


def segment_text(text: str):
    segments = []
    doc = nlp(text)
    subtrees = extract_attribution_indices(doc, nlp)
    if subtrees:
        for subtree in subtrees:
            segments.append(" ".join([t.text for t in subtree]))
    return segments


def dvmp_dataset_creation(num_samples, dest_path, max_num_phrases=3):
    prompts = []
    prompts_num_modifiers_prompt = []
    prompts_num_phrases_per_prompt = []

    while len(prompts) < num_samples:
        num_phrases = random.choice([i for i in range(1, max_num_phrases + 1)])
        prompt, num_modifiers = generate_prompt(num_phrases)
        if prompt not in prompts:
            prompts_num_phrases_per_prompt.append(num_phrases)
            prompts_num_modifiers_prompt.append(num_modifiers)


            segments = segment_text(prompt)
            num_mods = sum([len(s.split(" ")[:-1]) for s in segments])
            if num_mods != num_modifiers:

                print(prompt, num_mods, num_modifiers)
            prompts.append(prompt)

    subjects = []

    # add subjects to each prompt
    docs = nlp.pipe(prompts)
    for doc in docs:
        subjects.append([token.text for token in doc if token.pos_ == 'NOUN'])

    # convert to df and save
    df = pd.DataFrame({'prompt': prompts, 'num_modifiers': prompts_num_modifiers_prompt, 'num_phrases': prompts_num_phrases_per_prompt, 'subjects': subjects})
    if dest_path:
        df.to_csv(dest_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a DVMP dataset.')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples to generate.')
    parser.add_argument('--dest_path', type=str, default='destination.csv', help='Destination CSV file path.')
    args = parser.parse_args()

    dvmp_dataset_creation(args.num_samples, args.dest_path)
