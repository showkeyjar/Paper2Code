import json
import os
from tqdm import tqdm
from utils import extract_planning, content_to_json, print_response
import copy
import sys
from transformers import AutoTokenizer
from llm_client import LLMClient

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)

parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                    help="模型名称或路径，格式为 'provider:model_name'，例如 'deepseek:deepseek-chat' 或 'ollama:llama3'")
parser.add_argument('--provider', type=str, default="vllm",
                    choices=["vllm", "deepseek", "ollama"],
                    help="LLM 提供商，支持 vllm, deepseek, ollama")
parser.add_argument('--local_vllm',type=bool, default=False,
                    help="是否使用本地 vLLM")
parser.add_argument('--api_key', type=str, default="",
                    help="API 密钥（DeepSeek 需要）")
parser.add_argument('--base_url', type=str, default="",
                    help="API 基础 URL（DeepSeek 或 Ollama 需要）")
parser.add_argument('--tp_size', type=int, default=2,
                    help="张量并行大小（仅 vLLM 需要）")
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--max_tokens', type=int, default=4096,
                    help="最大生成 token 数")
parser.add_argument('--max_model_len', type=int, default=128000,
                    help="最大模型上下文长度（仅 vLLM 需要）")

parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format

parser.add_argument('--output_dir',type=str, default="")

args    = parser.parse_args()

paper_name = args.paper_name

# 解析模型名称和提供者
if ':' in args.model_name:
    provider, model_name = args.model_name.split(':', 1)
else:
    provider = args.provider
    model_name = args.model_name

tp_size = args.tp_size
max_model_len = args.max_model_len
temperature = args.temperature
max_tokens = args.max_tokens
local_vllm = args.local_vllm

# 初始化LLM客户端
llm_client = LLMClient(
    model_name=model_name,
    provider=provider,
    api_key=args.api_key or os.getenv("DEEPSEEK_API_KEY"),
    base_url=args.base_url,
    tp_size=tp_size,
    max_model_len=max_model_len,
    local_vllm=local_vllm
)

paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path

output_dir = args.output_dir
    
if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)

with open(f'{output_dir}/planning_config.yaml') as f: 
    config_yaml = f.read()

context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

# 0: overview, 1: detailed, 2: PRD
if os.path.exists(f'{output_dir}/task_list.json'):
    with open(f'{output_dir}/task_list.json') as f:
        task_list = json.load(f)
else:
    task_list = content_to_json(context_lst[2])

if 'Task list' in task_list:
    todo_file_lst = task_list['Task list']
elif 'task_list' in task_list:
    todo_file_lst = task_list['task_list']
elif 'task list' in task_list:
    todo_file_lst = task_list['task list']
else:
    print(f"[ERROR] 'Task list' does not exist. Please re-generate the planning.")
    sys.exit(0)

if 'Logic Analysis' in task_list:
    logic_analysis = task_list['Logic Analysis']
elif 'logic_analysis' in task_list:
    logic_analysis = task_list['logic_analysis']
elif 'logic analysis' in task_list:
    logic_analysis = task_list['logic analysis']
else:
    print(f"[ERROR] 'Logic Analysis' does not exist. Please re-generate the planning.")
    sys.exit(0)

done_file_lst = ['config.yaml']
logic_analysis_dict = {}
for desc in logic_analysis:
    logic_analysis_dict[desc[0]] = desc[1]

analysis_msg = [
    {"role": "system", "content": f"""You are an expert researcher, strategic analyzer and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in {paper_format} format, an overview of the plan, a design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml". 

Your task is to conduct a comprehensive logic analysis to accurately reproduce the experiments and methodologies described in the research paper. 
This analysis must align precisely with the paper’s methodology, experimental setup, and evaluation criteria.

1. Align with the Paper: Your analysis must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present your analysis in a logical, well-organized, and actionable format that is easy to follow and implement.
3. Prioritize Efficiency: Optimize the analysis for clarity and practical implementation while ensuring fidelity to the original experiments.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. REFER TO CONFIGURATION: Always reference settings from the config.yaml file. Do not invent or assume any values—only use configurations explicitly provided.
     
"""}]

def get_write_msg(todo_file_name, todo_file_desc):
    
    draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
    if len(todo_file_desc.strip()) == 0:
        draft_desc = f"Write the logic analysis in '{todo_file_name}'."

    write_msg=[{'role': 'user', "content": f"""## Paper
{paper_content}

-----

## Overview of the plan
{context_lst[0]}

-----

## Design
{context_lst[1]}

-----

## Task
{context_lst[2]}

-----

## Configuration file
```yaml
{config_yaml}
```
-----

## Instruction
Conduct a Logic Analysis to assist in writing the code, based on the paper, the plan, the design, the task and the previously specified configuration file (config.yaml). 
You DON'T need to provide the actual code yet; focus on a thorough, clear analysis.

{draft_desc}

-----

## Logic Analysis: {todo_file_name}"""}]
    return write_msg



model_name = args.model_name

# Initialize tokenizer with offline support
tokenizer = None
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",  # Use a small default model
        local_files_only=True,
        trust_remote_code=True
    )
except Exception as e:
    print(f"Warning: Could not load tokenizer: {e}")
    print("Will continue without tokenizer. Some features may be limited.")

def run_llm(msg):
    return llm_client.generate(
        messages=msg,
        temperature=temperature,
        max_tokens=max_tokens
    )

if "Qwen" in model_name:
    llm = LLM(model=model_name, 
            tensor_parallel_size=tp_size, 
            max_model_len=max_model_len,
            gpu_memory_utilization=0.95,
            trust_remote_code=True, enforce_eager=True, 
            rope_scaling={"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"})
    sampling_params = SamplingParams(temperature=temperature, max_tokens=131072)

elif "deepseek" in model_name:
    llm = LLM(model=model_name, 
              tensor_parallel_size=tp_size, 
              max_model_len=max_model_len,
              gpu_memory_utilization=0.95,
              trust_remote_code=True, enforce_eager=True)
artifact_output_dir=f'{output_dir}/analyzing_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

for todo_file_name in tqdm(todo_file_lst):
    responses = []
    trajectories = copy.deepcopy(analysis_msg)

    current_stage=f"[ANALYSIS] {todo_file_name}"
    print(current_stage)
    if todo_file_name == "config.yaml":
        continue
    
    if todo_file_name not in logic_analysis_dict:
        # print(f"[DEBUG ANALYSIS] {paper_name} {todo_file_name} is not exist in the logic analysis")
        logic_analysis_dict[todo_file_name] = ""
        
    instruction_msg = get_write_msg(todo_file_name, logic_analysis_dict[todo_file_name])
    trajectories.extend(instruction_msg)
        
    completion = run_llm(trajectories)
    
    # response
    completion_json = {
        'text': completion
    }

    # print and logging
    print_response(completion_json, is_llm=True)

    responses.append(completion_json)
    
    # trajectories
    trajectories.append({'role': 'assistant', 'content': completion})


    # save
    with open(f'{artifact_output_dir}/{todo_file_name}_simple_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(completion)

    done_file_lst.append(todo_file_name)

    # save for next stage(coding)
    todo_file_name = todo_file_name.replace("/", "_") 
    with open(f'{output_dir}/{todo_file_name}_simple_analysis_response.json', 'w', encoding='utf-8') as f:
        json.dump(responses, f)

    with open(f'{output_dir}/{todo_file_name}_simple_analysis_trajectories.json', 'w', encoding='utf-8') as f:
        json.dump(trajectories, f)
