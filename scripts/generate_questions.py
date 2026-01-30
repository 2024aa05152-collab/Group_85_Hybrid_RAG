import json
import random
import uuid
import os
import time
import re
from typing import List, Dict
from huggingface_hub import InferenceClient
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class QuestionGenerator:
    """
    A robust Q&A generator that handles rate limits, resumes progress,
    and ensures diverse question types across a corpus.
    """
    def __init__(self, model="meta-llama/Llama-3.1-8B-Instruct"):
        self.model = model
        self.categories = ["factual", "comparative", "inferential", "multi-hop"]
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("CRITICAL: HF_TOKEN not found in .env file.")
        
        self.client = InferenceClient(api_key=hf_token)

    def generate_single_qa(self, chunks: List[Dict], category: str) -> Dict:
        """Calls HF API to generate a single JSON Q&A pair."""
        context_text = "\n\n".join([f"Source: {c['text']}" for c in chunks])
        
        prompt = f"""Task: Generate ONE {category} question based on the context provided.
        Context: {context_text}
        
        Return ONLY a JSON object with these keys:
        {{
            "question": "string",
            "answer": "string",
            "question_type": "{category}"
        }}"""

        try:
            time.sleep(2.0) 
            response = self.client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that only outputs valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
            )
            
            content = response.choices[0].message.content.strip()
            # Remove potential markdown formatting
            content = re.sub(r'```json\s*|```', '', content).strip()
            raw_qa = json.loads(content)
            
            return {
                "id": f"q_{uuid.uuid4().hex[:6]}",
                "question": raw_qa.get("question"),
                "answer": raw_qa.get("answer"),
                "question_type": category,
                "source_chunk_id": chunks[0].get("chunk_id", "unknown"),
                "source_url": chunks[0].get("url"),
                "source_title": chunks[0].get("title")
            }
        except Exception as e:
            if "402" in str(e):
                print(f"Quota exceeded. Skipping this chunk for now.")
            else:
                print(f"Error during generation: {e}")
            return None

    def run_pipeline(self, chunks_path: str, output_path: str, target_count=100):
        """Manages the full flow: load, check progress, generate, and save."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load Chunks
        if not os.path.exists(chunks_path):
            print(f"Error: {chunks_path} not found.")
            return

        with open(chunks_path, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)

        existing_data = []
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        
        current_count = len(existing_data)
        if current_count >= target_count:
            print(f"Already reached target: {current_count} questions exist.")
            return existing_data

        used_ids = {q.get("source_chunk_id") for q in existing_data}
        available_chunks = [c for c in all_chunks if c.get("chunk_id") not in used_ids]
        
        needed = target_count - current_count
        print(f"Resuming: {current_count}/{target_count}. Generating {needed} more...")

        # Sequential Generation
        # We use a step to ensure we spread the 100 questions across all available chunks
        step = max(1, len(available_chunks) // needed)
        
        new_items = []
        for i in range(needed):
            idx = (i * step) % len(available_chunks)
            chunk = available_chunks[idx]
            category = self.categories[(current_count + i) % len(self.categories)]
            
            # Context setup (Multi-hop needs 2 chunks)
            context = [chunk]
            if category == "multi-hop":
                context.append(random.choice(all_chunks))

            print(f"[{i+current_count+1}/{target_count}] Generating {category}...")
            qa_pair = self.generate_single_qa(context, category)
            
            if qa_pair:
                new_items.append(qa_pair)
                # SAVE IMMEDIATELY (Checkpointing)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data + new_items, f, indent=4, ensure_ascii=False)
            else:
                print("Generation failed. Waiting 10s...")
                time.sleep(10)

        print(f"Success! Total questions: {len(existing_data + new_items)}")

if __name__ == "__main__":
    # Defining our file paths
    CHUNKS_FILE = 'data/corpus_chunks.json'
    OUTPUT_FILE = 'data/questions_100.json'
    
    generator = QuestionGenerator()
    generator.run_pipeline(CHUNKS_FILE, OUTPUT_FILE, target_count=100)