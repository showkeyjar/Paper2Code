import os
from typing import Dict, List, Optional, Union, Any
import json
import requests
from openai import OpenAI
from transformers import AutoTokenizer

try:
    from vllm import LLM as VLLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # 定义虚拟类以在没有 vllm 时防止错误
    class VLLM:
        def __init__(self, **kwargs):
            raise ImportError("vllm is not installed. Please install it with 'pip install vllm' or use a different provider.")
    
    class SamplingParams:
        def __init__(self, **kwargs):
            pass

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
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name if provider == "vllm" else "gpt2",
                trust_remote_code=True,
                local_files_only=True,  # 只使用本地文件
                revision="main"
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            print("Will continue without tokenizer. Some features may be limited.")
            self.tokenizer = None
        
        if self.provider == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "vllm is not installed. Please install it with 'pip install vllm' or use a different provider.\n"
                    "You can install it with: pip install vllm"
                )
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
    
    def _generate_ollama(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        """使用Ollama API生成文本"""
        url = f"{self.base_url}/api/chat"
        
        # 确保消息格式正确
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            else:
                formatted_messages.append({"role": "user", "content": str(msg)})
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": False,  # 确保使用非流式响应
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            print(f"Sending request to Ollama API...")
            print(f"URL: {url}")
            print(f"Model: {self.model_name}")
            print(f"Messages: {formatted_messages}")
            
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            # 调试信息
            print(f"Ollama response status: {response.status_code}")
            response_text = response.text
            print(f"Ollama response content (first 500 chars): {response_text[:500]}")
            
            # 尝试解析JSON响应
            try:
                response_data = response.json()
                print(f"Parsed response data: {response_data}")
                
                # 处理不同的响应格式
                if isinstance(response_data, dict):
                    if "message" in response_data and "content" in response_data["message"]:
                        return response_data["message"]["content"]
                    elif "content" in response_data:
                        return response_data["content"]
                    else:
                        print("Unexpected response format, returning full response")
                        return str(response_data)
                else:
                    return str(response_data)
                    
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Raw response: {response_text}")
                return response_text
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Ollama API: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nStatus code: {e.response.status_code}"
                try:
                    error_msg += f"\nResponse: {e.response.text}"
                except:
                    pass
            print(error_msg)
            raise Exception(error_msg) from e

    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.tokenizer.encode(text))
