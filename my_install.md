# 安装过程记录

供参考


## 部署本地 vllm (可选)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


## 部署docker版 vllm（可选）

pip install modelscope

modelscope download --local_dir /mnt/nas13/llm_models/deepseek-70b --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B

modelscope download --local_dir /mnt/nas13/llm_models/Qwen3-30B --model Qwen/Qwen3-30B-A3B


sudo docker run -itd --restart=always --name vllm_ds70 \
-v /mnt/nas13/llm_models:/models \
-p 18005:8000 \
--gpus all \
--ipc=host \
vllm/vllm-openai:latest \
--dtype bfloat16 \
--served-model-name DeepSeek-R1-Distill-Llama-70B \
--model "/models/deepseek-70b" \
--gpu-memory-utilization 0.9 \
--tensor-parallel-size 8 \
--max-model-len 30000 \
--api-key token-abc123


nohup python -u -m vllm.entrypoints.openai.api_server --port 8081 --model /mnt/nas13/llm_models/deepseek-70b --trust-remote-code --tensor-parallel-size 1 2>&1 | tee api_server.log &


## 将 pdf转换为json 格式

cd ~/s2orc-doc2json
python doc2json/grobid2json/process_pdf.py -i input_dir/carbone2ec.pdf -t temp_dir/ -o output_dir/
