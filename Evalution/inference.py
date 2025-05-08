import argparse
import time
import json

# from vllm import LLM, SamplingParams
from yalis import ModelConfig, InferenceConfig, LLMEngine, print_rank0
import torch
import os
from transformers import AutoTokenizer
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM with command line arguments."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model type to use."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8000,
        help="Maximum length of generation.",
    )
    parser.add_argument(
        "--gpu", type=int, default=1, help="Number of GPUs to use."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output file path."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="input file path."
    )
    parser.add_argument(
        "--attn_backend",
        type=str,
        required=True,
        help="attn_backend - [sdpa, flash, thresh]",
    )
    parser.add_argument("--warmup", type=int, help="warmup steps")
    parser.add_argument("--percentile", type=float, help="percentile")

    args = parser.parse_args()
    return args


# Process output to split blocks and count words
def process_output(output: str, retain_perc: float) -> dict:
    blocks = output.split("#*#")
    word_count = len(output.split())
    return {
        "blocks": blocks,
        "word_count": word_count,
        "retain_perc": retain_perc,
    }


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def save_to_json(data: list, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_inputs(filename: str) -> list:
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Combine inputs, results and word counts and save them
def process_and_save_results(
    inputs: list, results: list, filename: str
) -> None:
    combined = []
    for input_data, result_data in zip(inputs, results):
        combined.append(
            {
                "input": input_data["prompt"],
                "checks_once": input_data["checks_once"],
                "checks_range": input_data["checks_range"],
                "checks_periodic": input_data["checks_periodic"],
                "type": input_data["type"],
                "number": input_data["number"],
                "output_blocks": result_data["blocks"],
                "word_count": result_data[
                    "word_count"
                ],  # Adding word count here
                "retain_perc": result_data["retain_perc"],  
            }
        )
    save_to_json(combined, filename)


args = parse_args()

# input_file = '/home/yuhao/THREADING-THE-NEEDLE/Dataset/Dataset_short.json'
inputs = load_inputs(args.input_file)

print(f"Loaded {len(inputs)} inputs from {args.input_file}")

batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

# Get max prompt length
max_prompt_length = max(
    len(tokenizer.encode(input_data["prompt"])) for input_data in inputs
)

inferece_config = InferenceConfig(
    batch_size=batch_size,
    max_length_of_generated_sequences=max_prompt_length + args.max_length,
    top_p=0.95,
    temperature=0.95,
    tp_dims=(args.gpu, 1, 1),
    attention_backend=args.attn_backend,
    threshold_percentile=args.percentile,
    num_warmup_steps=args.warmup,
    use_paged_kv_caching=False,
    prestore_kv_cache=True,
)

model_config = ModelConfig(model_name=args.model, precision="fp16")

engine = LLMEngine(model_config=model_config, inference_config=inferece_config)

results = []
retain_perc = 0
num_batches = 0
for i in range(0, len(inputs), batch_size):
    batch_inputs = inputs[i : i + batch_size]
    num_batches += 1

    prompts = [input_data["prompt"] for input_data in batch_inputs]

    start_time = time.time()
    outputs, metrics, _ = engine.generate(
        prompts, tokens_to_generate=args.max_length
    )
    end_time = time.time()
    inference_time = end_time - start_time
    print(
        f"Inference time for batch {i//batch_size + 1}: {inference_time:.2f} seconds"
    )

    if not math.isnan(metrics["BatchRetainPercentage"]):
        retain_perc += metrics["BatchRetainPercentage"]
    elif args.attn_backend == "thresh":
        raise Exception("This should not happen")

    detokenized_outputs = tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )
    # Truncate the outputs till the first occurrence of '*** finished'
    for retain_perc_prompt, text in zip(
        metrics["RetainPercentage"], detokenized_outputs
    ):
        if "*** finished" in text:
            text = text.split("*** finished")[0]
        results.append(process_output(text, retain_perc_prompt))


retain_perc /= num_batches
print(f"Average retain percentage: {retain_perc:.2f}")

process_and_save_results(inputs, results, args.output_file)
print(f"\nSaved result to {args.output_file}")


# sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=args.max_length, seed=42, repetition_penalty = 1.005)
# sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=args.max_length, seed=6211027, stop = '*** finished')
#
# prompts = [input_data['prompt'] for input_data in inputs]
#
## Setting up the LLM with the specified number of GPUs and model
# llm = LLM(model=args.model, tensor_parallel_size=args.gpu, gpu_memory_utilization=0.95)
#
# start_time = time.time()
# outputs = llm.generate(prompts, sampling_params)
# inference_time = time.time() - start_time
# print(f"Inference time: {inference_time:.2f} seconds")
#
# results = [process_output( input['prefix']+ output.outputs[0].text) for output, input in zip(outputs,inputs)]
#
# process_and_save_results(inputs, results, args.output_file)
# print(f"\nSaved result to {args.output_file}")
