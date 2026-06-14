import json
import pandas as pd
import ast
from datasets import Dataset
import re

# def format_sample(data, i):
#   sample = data[i]['new_label']
#   text = data[i]['review']
#   aspects = []
#   aspect_sentiment_pairs = []
#   aspect_opinion_pairs = []
#   triplets = []
#   for x in sample:
#       sentiment = x['sentiment']

#       for triplet in x['triples']:
#           aspect = triplet['aspect']
#           opinion = triplet['opinion']

#           if not aspect:
#               aspect = "<IA>"
#           else:
#               aspects.append(aspect)

#           if not opinion:
#               opinion = "<IO>"

#           # lấy as pair
#           as_pair = (aspect, sentiment)
#           aspect_sentiment_pairs.append(as_pair)

#           # lấy aspect opinion pair
#           ao_pair = (aspect, opinion)
#           aspect_opinion_pairs.append(ao_pair)

#           # lấy triplet
#           triplet = (aspect, opinion, sentiment)
#           triplets.append(triplet)

#   sample_new_format = {
#   "index_sentence": data[i]['Index_sentence'],
#   "text": text,
#   "aspects": aspects,
#   "aspect_sentiment_pairs": aspect_sentiment_pairs,
#   "aspect_opinion_pairs": aspect_opinion_pairs,
#   "triplets": triplets
#   }
#   return sample_new_format


def format_sample(data, i):
    aspects = []
    aspect_sentiment_pairs = []
    aspect_opinion_pairs = []
    triplets = []

    parts = data.iloc[i].label.split("####")
    if len(parts) < 2:
        return {
        "text": parts[0],
        "aspects": aspects,
        "aspect_sentiment_pairs": aspect_sentiment_pairs,
        "aspect_opinion_pairs": aspect_opinion_pairs,
        "triplets": triplets
        }

    tokens = parts[0].split()
    raw_triplets = ast.literal_eval(parts[1])
    for raw_triplet in raw_triplets:
        a, o, p = raw_triplet
        if a:
            aspect = " ".join(tokens[a[0]: a[-1]+1])
        else:
            aspect = "<IA>"

        if o:
            opinion = " ".join(tokens[o[0]: o[-1]+1])
        else:
            opinion = "<IO>"

        sentiment = p

        aspects.append(aspect)
        aspect_sentiment_pairs.append((aspect, sentiment))
        aspect_opinion_pairs.append((aspect, opinion))
        triplets.append((aspect, opinion, sentiment))

    sample_new_format = {
    "text": parts[0],
    "aspects": aspects,
    "aspect_sentiment_pairs": aspect_sentiment_pairs,
    "aspect_opinion_pairs": aspect_opinion_pairs,
    "triplets": triplets
    }
    return sample_new_format

def format_data(data):
    N = len(data)
    samples = []
    for i in range(N):
        samples.append(format_sample(data, i))
    
    df = pd.DataFrame(samples)
    return df

def get_rows(formatted_sample, tasks = ["ate", "aooe", "aope", "atsc", "aspe", "aoste"]):
    rows = []
    text = formatted_sample['text']
    aspects = formatted_sample['aspects']

    task = "ate"
    if task in tasks:
        Input = text
        if not aspects:
            Output = "<IA>"
        else:
            Output = " ## ".join(aspects)

        rows.append({"Task": task, "Input": Input, "Output": f"[{task}] {Output}"})

    task = "aooe"
    if task in tasks:
        aspect_opinion_pairs = formatted_sample['aspect_opinion_pairs']
        aspect_key = {}

        for a, o in aspect_opinion_pairs:
            if a not in aspect_key:
                aspect_key[a] = []
            aspect_key[a].append(o)

        # loại <IA> nếu nó không có opinion
        if '<IA>' in aspect_key and not aspect_key['<IA>']:
            aspect_key.pop('<IA>')

        if len(aspect_key) > 0:
            for a, opinions in aspect_key.items():
                Input = f"{text} ## Aspect: {a}"
                Output = " ## ".join(opinions) if opinions else "<IO>"
                rows.append({"Task": task, "Input": Input, "Output": f"[{task}] {Output}"})

    # task aspe
    task = "aope"
    if task in tasks:
        if len(aspect_opinion_pairs) == 0:
            Input = text
            Output="<IA> $ <IO>"
        else:
            Input = text
            Output = []
            for a, o in aspect_opinion_pairs:
                Output.append(f"{a} $ {o}")
            Output = " ## ".join(Output)
        rows.append({"Task": task, "Input": Input, "Output": f"[{task}] {Output}"})

    # task atsc 
    task = "atsc"
    if task in tasks:
        aspect_sentiment_pairs = formatted_sample['aspect_sentiment_pairs']
        aspect_key = {}
        # Chỉ lấy sentiment đầu tiên cho mỗi aspect, bỏ qua <IA>
        for a, s in aspect_sentiment_pairs:
            if a != '<IA>' and a not in aspect_key:
                aspect_key[a] = s

        if len(aspect_key) > 0:
            for a, s in aspect_key.items():
                Input = f"{text} ## Aspect: {a}"
                Output = s
                rows.append({"Task": task, "Input": Input, "Output": f"[{task}] {Output}"})

    # task aspe
    task = "aspe"
    if task in tasks:
        if len(aspect_sentiment_pairs) == 0:
            Input = text
            Output="<IA> $ none"
        else:
            Input = text
            Output = []
            i = 0
            for a, s in aspect_sentiment_pairs:
                if a == "<IA>":
                    a = f"<IA>"
                    i+=1
                Output.append(f"{a} $ {s}")
            Output = " ## ".join(Output)
        rows.append({"Task": task, "Input": Input, "Output": f"[{task}] {Output}"})

    # triplet
    task = "aoste"
    if task in tasks:
        Input = text
        triplets = formatted_sample['triplets']
        if len(triplets) == 0:
            Output="<IA> $ <IO> $ none"
        else:
            Output = []
            for a, o, s in triplets:
                Output.append(f"{a} $ {o} $ {s}")
            Output = " ## ".join(Output)
        rows.append({"Task": task, "Input": Input, "Output": f"[{task}] {Output}"})
    return rows


def create_data_with_task(df, tasks = ["ate", "aooe", "aope", "atsc", "aspe", "aoste"]):
    N = len(df)
    rows = []
    for i in range(N):
        sample = {
            "text": df.iloc[i]['text'],
            "aspects": ast.literal_eval(df.iloc[i]['aspects']),
            "aspect_sentiment_pairs": ast.literal_eval(df.iloc[i]['aspect_sentiment_pairs']),
            "aspect_opinion_pairs": ast.literal_eval(df.iloc[i]['aspect_opinion_pairs']),
            "triplets": ast.literal_eval(df.iloc[i]['triplets'])
        }

        rows.extend(get_rows(sample, tasks))
    
    return pd.DataFrame(rows)


def clean_text(text: str) -> str:
    # bỏ dấu câu nhưng giữ chữ, số, khoảng trắng, dấu tiếng Việt, [, ]
    return re.sub(r"[^\w\s\[\]]", "", text, flags=re.UNICODE)

def remove_punctuation(text: str) -> str:
    """
    Loại bỏ dấu câu trong phần input và output của các ví dụ.
    Giữ nguyên cấu trúc ví dụ.
    """

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        if line.strip().startswith(("input:", "output:")):
            prefix, content = line.split(":", 1)
            content = clean_text(content)
            cleaned_lines.append(f"{prefix}:{content}")
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def get_dataset_for_training(df, instruction_handler, allow_punctuation = True):
    N = len(df)
    Input = []
    Output = []

    for i in range(N):
        task = df.iloc[i].Task
        text = df.iloc[i].Input
        output = df.iloc[i].Output
        instruction = instruction_handler.apply_instruction(text, task)

        if not allow_punctuation:
            instruction = remove_punctuation(instruction)
            output = clean_text(output)

        Input.append(instruction)
        Output.append(output)
    
    return Dataset.from_dict({"input": Input, "output": Output})