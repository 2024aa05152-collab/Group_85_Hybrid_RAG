import os
import json
import tiktoken
import uuid
from wikipedia_loader import WikipediaLoader
from text_cleaner import TextCleaner

class WikipediaChunker:
    def __init__(self, chunk_size=400, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text):
        return len(self.encoding.encode(text))
    
    def create_chunks(self, text, url, title):
        tokens = self.encoding.encode(text)
        chunks = []
        
        # Sliding window using token counts
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Ensure chunk is substantial
            if len(chunk_tokens) < 100 and len(chunks) > 0:
                continue 

            chunks.append({
                "chunk_id": str(uuid.uuid4())[:8], # Unique ID for each piece
                "url": url,
                "title": title,
                "text": chunk_text,
                "token_count": len(chunk_tokens)
            })
        return chunks

def run_pipeline():
    loader = WikipediaLoader()
    cleaner = TextCleaner()
    chunker = WikipediaChunker(chunk_size=400, chunk_overlap=50)
    
    all_urls = []

    # Load Fixed URLs (Category Dict format)
    if os.path.exists('data/fixed_urls.json'):
        with open('data/fixed_urls.json', 'r') as f:
            data = json.load(f)
            for category, urls in data.items():
                all_urls.extend(urls)
    # Load Random URLs (List format)
    if os.path.exists('data/random_urls.json'):
        with open('data/random_urls.json', 'r') as f:
            data = json.load(f)
            for item in data:

                if isinstance(item, str):
                    all_urls.append(item)
                elif isinstance(item, dict):
                    all_urls.append(item.get('url'))

    all_chunks = []
    print(f"Total unique URLs to process: {len(set(all_urls))}")

    # Process each URL
    for i, url in enumerate(list(set(all_urls))):
        print(f"[{i+1}/{len(all_urls)}] Processing: {url}")
        
        page_data = loader.fetch_content(url)
        if page_data["success"]:
            clean_text = cleaner.clean_html(page_data["html_content"])
            
            if cleaner.validate_text_length(clean_text, min_words=200):
                chunks = chunker.create_chunks(clean_text, url, page_data["title"])
                all_chunks.extend(chunks)
        
    os.makedirs('data', exist_ok=True)
    with open('data/corpus_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)
    
    print(f"\nCreated {len(all_chunks)} chunks.")

if __name__ == "__main__":
    run_pipeline()