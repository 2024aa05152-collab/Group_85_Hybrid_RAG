import tiktoken
import json
from typing import List, Dict, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk text into overlapping segments"""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap"""
        tokens = self.encoder.encode(text)
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            # Ensure we don't cut in the middle of a word if possible
            if end < len(tokens):
                # Try to find a sentence boundary
                chunk_text = self.encoder.decode(tokens[start:end])
                last_period = max(chunk_text.rfind('.'),
                                  chunk_text.rfind('!'),
                                  chunk_text.rfind('?'))

                if last_period > len(chunk_text) * 0.5:  # If period in second half
                    adjusted_end = start + len(self.encoder.encode(chunk_text[:last_period + 1]))
                    if adjusted_end > start:  # Ensure we make progress
                        end = adjusted_end

            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            # Create chunk with metadata
            chunk_data = {
                "chunk_id": f"{metadata['url']}_{chunk_id}",
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end,
                "url": metadata["url"],
                "title": metadata["title"],
                "chunk_index": chunk_id,
                "total_chunks": None  # Will be filled later
            }

            chunks.append(chunk_data)
            chunk_id += 1

            # Move start position with overlap
            start = end - self.chunk_overlap
            if start < end - 1:  # Ensure we make progress
                start = end - 1

        # Add total_chunks to each chunk
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)

        return chunks

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all documents into chunks"""
        all_chunks = []
        chunk_counter = 0

        for doc in documents:
            chunks = self.chunk_text(doc["cleaned_text"], {
                "url": doc["url"],
                "title": doc["title"]
            })

            for chunk in chunks:
                chunk["global_chunk_id"] = chunk_counter
                chunk_counter += 1
                all_chunks.append(chunk)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: Path):
        """Save chunks to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: Path) -> List[Dict[str, Any]]:
        """Load chunks from JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
        return chunks