import pandas as pd
import textwrap
import torch
import argparse
import sys
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

DEVICE = 'cuda'

prompt_template = """You are an assistant tasked with predicting emotion words masked as <mask> in a given self-disclosure text from social media. Predict the {num_labels} <mask> token(s) based on the context. 
Provide your answer in the format: [{example_format}]. The length of the list must be {num_labels}. Only include words describing emotions, and provide no extra text or reasoning.

Text:
{segment}

Answer:
"""

def generate_example_format(num_labels):
    return ", ".join([f"'insert emotion word for mask {i+1}'" for i in range(num_labels)])

def generate_batch_responses(data, model, tokenizer, device, template, output_path, batch_size=8):
    if 'output' not in data.columns:
        data['output'] = None

    try:
        existing_data = pd.read_csv(output_path)
        data.update(existing_data)
    except FileNotFoundError:
        pass

    progress_bar = tqdm(total=len(data), desc="Processing Rows", file=sys.stdout)

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        batch = batch[batch['output'].isnull()]

        if batch.empty:
            progress_bar.update(batch_size)
            continue

        prompts = [
            template.format(
                num_labels=row['number_of_labels'],
                example_format=generate_example_format(row['number_of_labels']),
                segment=row['segment']
            )
            for _, row in batch.iterrows()
        ]

        encoding = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=256,
                temperature=0.0
            )

        decoded_outputs = []
        for idx, response in enumerate(outputs):
            try:
                decoded = tokenizer.decode(response, skip_special_tokens=True)
                if "Answer:" in decoded:
                    decoded = decoded.split("Answer:")[1].strip()
                else:
                    decoded = tokenizer.decode(response[input_ids[idx].shape[0]:], skip_special_tokens=True)
                decoded_outputs.append("\n".join(textwrap.wrap(decoded.strip())))
            except Exception:
                decoded_outputs.append("Error decoding response")

        data.loc[batch.index, 'output'] = decoded_outputs
        data.to_csv(output_path, index=False)
        progress_bar.update(len(batch))

    progress_bar.close()
    data.to_csv(output_path, index=False)
    return data

def main():
    parser = argparse.ArgumentParser(description="Run emotion prediction using a causal LLM.")
    parser.add_argument('--token', required=True, help='HuggingFace authentication token.')
    parser.add_argument('--model', default='google/gemma-2-9b-it', help='Model name from HuggingFace.')
    parser.add_argument('--input', required=True, help='Path to input CSV file.')
    parser.add_argument('--output', required=True, help='Path to output CSV file.')
    parser.add_argument('--download_path', default='./', help='Local cache dir for model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation.')

    args = parser.parse_args()

    # Load model
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    print(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        use_auth_token=args.token,
        cache_dir=args.download_path
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    data = pd.read_csv(args.input)
    print(f"Loaded {len(data)} rows from {args.input}")

    generate_batch_responses(
        data=data,
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        template=prompt_template,
        output_path=args.output,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()