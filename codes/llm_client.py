import os
from typing import Dict, List, Optional, Union
import json
import requests
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM as VLLM, SamplingParams

class LLMClient:
    """统一管理不同LLM提供商的客户端"""
    
    def __init__(self, model_name: str, provider: str = "vllm", **kwargs):
        """
        初始化LLM客户端
        
        Args:
            model_name: 模型名称或路径
            provider: 提供商，支持 'vllm', 'deepseek', 'ollama'
            **kwargs: 其他参数，如api_key, base_url等
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name if provider == "vllm" else "gpt2")
        
        if self.provider == "vllm":
            self.client = VLLM(
                model=model_name,
                tensor_parallel_size=kwargs.get("tp_size", 1),
                max_model_len=kwargs.get("max_model_len", 4096)
            )
        elif self.provider == "deepseek":
            self.client = OpenAI(
                api_key=kwargs.get("api_key", os.getenv("DEEPSEEK_API_KEY")),
                base_url=kwargs.get("base_url", "https://api.deepseek.com/v1")
            )
        elif self.provider == "ollama":
            self.base_url = kwargs.get("base_url", "http://localhost:11434")
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            temperature: 温度参数
            max_tokens: 最大生成token数
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        if self.provider == "vllm":
            return self._generate_vllm(messages, temperature, max_tokens, **kwargs)
        elif self.provider == "deepseek":
            return self._generate_deepseek(messages, temperature, max_tokens, **kwargs)
        elif self.provider == "ollama":
            return self._generate_ollama(messages, temperature, max_tokens, **kwargs)
    
    def _generate_vllm(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """使用vLLM生成文本"""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # 将消息转换为提示
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.client.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def _generate_deepseek(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """使用DeepSeek API生成文本"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
    
    def _generate_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """使用Ollama API生成文本"""
        # Ollama的API格式与OpenAI略有不同
        url = f"{self.base_url}/api/chat"
        
        # 转换消息格式
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        
        data = {
            "model": self.model_name,
            "messages": ollama_messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.tokenizer.encode(text))
