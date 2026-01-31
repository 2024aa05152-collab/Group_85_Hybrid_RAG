import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

class ResponseGenerator:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.cache_dir = os.getenv("HF_HOME", "./data/hf_cache")
        self.hf_token = os.getenv("HF_TOKEN")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=self.hf_token, cache_dir=self.cache_dir
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=self.hf_token,
            cache_dir=self.cache_dir,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype="auto", 
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        # Build the Text Generation Pipeline
        self.gen_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate(self, query: str, context_chunks: list) -> str:
        # Combine retrieved chunks into one context block
        context_text = "\n\n".join([
            f"--- Source {i+1} ---\n{c.get('text', '')}" 
            for i, c in enumerate(context_chunks)
        ])
        
        # Use Qwen's specific ChatML format for better accuracy
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the query. If the answer is not in the context, say you don't know. Cite your sources (e.g. [Source 1])."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        outputs = self.gen_pipeline(
            prompt, 
            max_new_tokens=512, 
            temperature=0.1, 
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return outputs[0]['generated_text'][len(prompt):].strip()