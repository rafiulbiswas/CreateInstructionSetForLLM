import json
import pandas as pd
from datasets import Dataset
from typing import List, Dict
import argparse

def prepare_dataset_llama3(dataset, instructions=None):
    # Convert pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(dataset)
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{}<|eot_id|><|start_header_id|>user<|end_header_id|>{}: {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>"""
    EOS_TOKEN = '<|end_of_text|>'
    
    def formatting_prompts_func(examples):
        sys_prompt = "You are a social media expert providing accurate analysis and insights on various types of text data."  
        instructions_list = instructions

        inputs = examples["text"]
        outputs = examples["Emotion_Label"]
        texts = []

        for input, output in zip(inputs, outputs):
            for instruction in instructions_list:
                text = prompt.format(sys_prompt, instruction, input, output) + EOS_TOKEN
                texts.append({"instruction": instruction, "text": input, "label": output})
        return {"output": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
    return dataset

def load_instructions(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)
    
def write_dataset_to_jsonl(dataset, output_file):
    """
    Writes the data from a Dataset instance to a JSONL file.
    """
    with open(output_file, 'w', encoding="utf-8") as file:
        for record in dataset["output"]:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process datasets based on instructions.")
    parser.add_argument('--instructions', type=str, required=True, help="Path to the instructions JSON file")
    parser.add_argument('--input_data', type=str, required=True, help="Path to the datasets CSV file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSONL file")

    # Parse the arguments
    args = parser.parse_args()

    # Load instructions
    instructions = load_instructions(args.instructions)

    # Load dataset
    dataset = pd.read_csv(args.input_data, encoding="utf-8")

    # Process dataset
    dataset = prepare_dataset_llama3(dataset, instructions)

    # Write to JSONL
    write_dataset_to_jsonl(dataset, args.output_file)

if __name__ == "__main__":
    main()
