import json
import random
import sys
import base64

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tvm
from datasets import load_dataset
from tvm import relax
from tvm.contrib import tvmjs
from tvm.runtime import Device, Module, Object, ShapeTuple
from tvm.runtime.relax_vm import VirtualMachine

from mlc_llm import MLCEngine
from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.interface.help import HELP
from mlc_llm.protocol.mlc_chat_config import MLCChatConfig
from mlc_llm.serve import data, engine_utils
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.support.auto_device import detect_device
from mlc_llm.support.style import green, red
from mlc_llm.tokenizers import Tokenizer

prompt_phi_3_5_v_few_shot = """Question:
Which of the following is the body cavity that contains the pituitary gland?
Options:
A. Abdominal
B. Cranial
C. Pleural
D. Spinal
Answer: B
Question:
Where was the most famous site of the mystery cults in Greece?
Options:
A. Ephesus
B. Corinth
C. Athens
D. Eleusis
Answer: D

"""

prompt_phi_3_5_v_zero_shot = """"""

def encode_image(image):
    rgb_image = image
    buffer = BytesIO()
    rgb_image.save(buffer, format="PNG")
    buffer.seek(0)
    image_str = base64.b64encode(buffer.read()).decode('utf-8')
    return image_str

def construct_prompt_mmmu(ex, prompt_prefix=prompt_phi_3_5_v_zero_shot):
    overall_prompt = prompt_prefix
    if 'question' in ex:
        overall_prompt += ex['question'] + "\n"
    if 'options' in ex:
        options = eval(ex['options'])
        for oi, option in enumerate(options):
            overall_prompt += f"{chr(oi+65)}: {option}\n"
    overall_prompt += "Answer: "
    return overall_prompt

def eval_mmmu(model, engine: MLCEngine, prompt=prompt_phi_3_5_v_zero_shot, temperature=0.0):
    slices = ["Accounting"]
    slice_correct = []
    slice_total = []
    for si, sl in enumerate(slices):
        ds = load_dataset("MMMU/MMMU", sl)
        slice_correct_here = 0
        slice_total_here = 0
        for exi in range(len(ds['validation'])):
            ex = ds['validation'][exi]
            preproc_ex = construct_prompt_mmmu(ex, prompt_phi_3_5_v_zero_shot)
            base64_image = encode_image(ex["image_1"])
            response = engine.chat.completions.create(
                messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type":"image_url", 
                                        "image_url": {"url":f"data:image/jpeg;base64,{base64_image}"}
                                    },
                                    {
                                        "type":"text", 
                                        "text":preproc_ex
                                    }
                                ]
                            }
                        ],
                model=model,
                stream=False,
                temperature=temperature,
            )
            ans = response.choices[0].message.content
            if ans.strip()[:1] == ds['validation'][exi]['answer'].strip():
                slice_correct_here += 1
                print("Correct")
            else:
                print("Wrong")
            slice_total_here += 1
        
        slice_correct.append(slice_correct_here)
        slice_total.append(slice_total_here)
        print(f"Slice: {sl} ; Statistics Below\nCorrect: {slice_correct_here}\nTotal: {slice_total_here}\nAccuracy: {slice_correct_here/slice_total_here}")

    overall_total = sum(slice_total)
    correct_total = sum(slice_correct)
    print(f"Overall Statistics Below\nCorrect: {correct_total}\nTotal: {overall_total}\nAccuracy: {correct_total/overall_total}")

def main():
    """The main function to start a DebugChat CLI"""
    parser = ArgumentParser("MLC LLM Correctness Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        help="An MLC model directory that contains `mlc-chat-config.json`",
        required=True,
    )
    parser.add_argument(
        "--model-lib",
        type=str,
        help="The full path to the model library file to use (e.g. a ``.so`` file).",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_compile"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--temperature",
        default=0.0,
        help="temperature for generation"
    )
    parsed = parser.parse_args()
    engine = MLCEngine(parsed.model, model_lib=parsed.model_lib)
    eval_mmmu(parsed.model, engine, temperature=parsed.temperature)
    
if __name__ == "__main__":
    main()
