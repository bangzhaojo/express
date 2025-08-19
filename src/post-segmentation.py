import re
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from huggingface_hub import login


def find_first_and_last_positions_or_return_A(A, is_start=False, is_end=False):
    B = ['.', '!', '?', 'Ġ.', 'Ġ!', 'Ġ?']
    first_position = None
    last_position = None

    for i, token in enumerate(A):
        if token in B:
            if first_position is None:
                first_position = i
            last_position = i

    if first_position == last_position or first_position is None:
        return A

    if is_start:
        return A[:last_position + 1]
    elif is_end:
        return A[first_position + 1:]
    else:
        return A[first_position + 1:last_position + 1]


def segmentation(text_tokens, token_limit=512, dist=256, ratio=(0.5, 0.5)):
    if len(text_tokens) <= token_limit:
        return [text_tokens]

    mask_positions = [i for i, token in enumerate(text_tokens) if token.strip() == '<mask>']

    if not mask_positions:
        return [text_tokens[:token_limit]]

    if mask_positions[-1] < token_limit:
        last_punctuation_index = None
        for i in range(mask_positions[-1] + 1, token_limit):
            if text_tokens[i] in {'.', '!', '?', 'Ġ.', 'Ġ!', 'Ġ?'}:
                last_punctuation_index = i
        return [text_tokens[:last_punctuation_index + 1] if last_punctuation_index else token_limit]

    elif len(mask_positions) == 1:
        start = max(0, mask_positions[0] - int(ratio[0] * (token_limit - 2)))
        end = min(len(text_tokens), mask_positions[0] + int(ratio[1] * (token_limit - 2)))
        return [find_first_and_last_positions_or_return_A(text_tokens[start:end], is_end=(end == len(text_tokens)))]

    else:
        groups = []
        current_group = []

        for i in range(len(mask_positions)):
            if not current_group:
                current_group.append(mask_positions[i])
            elif mask_positions[i] - current_group[-1] < dist:
                current_group.append(mask_positions[i])
            else:
                groups.append(current_group)
                current_group = [mask_positions[i]]
        if current_group:
            groups.append(current_group)

        centroids = [sum(group) / len(group) for group in groups]
        segments = []
        for c in centroids:
            start = max(0, round(c) - int(ratio[0] * (token_limit - 2)))
            end = min(len(text_tokens), round(c) + int(ratio[1] * (token_limit - 2)))
            segments.append(find_first_and_last_positions_or_return_A(text_tokens[start:end],
                                                                       is_start=(start == 0),
                                                                       is_end=(end == len(text_tokens))))
        return segments


def get_emotion_list(text):
    return [t.strip().replace("\"", "").replace("'", "") for t in text[1:-2].split(",")]


def get_number_of_masks(text):
    return sum(1 for j in text if j.strip() == "<mask>")


def process_posts(text, data, tokenizer, output_path, token_limit=512, dist=256, ratio=(0.5, 0.5)):
    i = 0
    posts, emotions, post_index, original_index, number_of_masks, new_labels, ids = [], [], [], [], [], [], []

    labels = data.emotions.to_list()

    with tqdm(total=len(text), desc="Processing Posts", unit="post") as pbar:
        while i < len(text):
            segments = segmentation(text[i], token_limit=token_limit, dist=dist, ratio=ratio)
            label_index = 0

            for s in segments:
                if len(segments) == 1:
                    new_labels.append(labels[i])
                else:
                    curr_label = get_emotion_list(labels[i])
                    masks = sum(1 for j in s if j.strip() == "<mask>")
                    new_labels.append(curr_label[label_index:label_index + masks])
                    label_index += masks

                posts.append(tokenizer.convert_tokens_to_string(s))
                post_index.append(i)
                emotions.append(data.emotions[i])
                number_of_masks.append(get_number_of_masks(s))
                original_index.append(data.iloc[i]['index'])
                ids.append(data.id[i])

            i += 1
            pbar.update(1)

    segmented_data = pd.DataFrame({
        'index': post_index,
        'original_index': original_index,
        'original_ids': ids,
        'segment': posts,
        'number_of_labels': number_of_masks,
        'labels': new_labels,
        'original_labels': emotions
    })

    segmented_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment and align masked posts with emotion labels.')
    parser.add_argument('--input', required=True, help='Path to input CSV file.')
    parser.add_argument('--output', required=True, help='Path to output CSV file.')
    parser.add_argument('--token', required=False, help='HuggingFace token to login if required.')
    parser.add_argument('--token_limit', type=int, default=512, help='Maximum number of tokens per segment.')
    parser.add_argument('--dist', type=int, default=256, help='Minimum distance between masks.')
    parser.add_argument('--ratio', nargs=2, type=float, default=(0.5, 0.5), help='Ratio around mask for segmentation.')

    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    tokenizer = AutoTokenizer.from_pretrained("AIMH/mental-roberta-large")
    data = pd.read_csv(args.input)

    # Drop broken row if needed
    if 30231 in data.index:
        data = data.drop(index=data.iloc[30231].name).reset_index(drop=True)

    text = data['aug'].str.replace('[MASK]', '<mask>', regex=False).apply(tokenizer.tokenize)

    process_posts(text, data, tokenizer, args.output, token_limit=args.token_limit, dist=args.dist, ratio=tuple(args.ratio))