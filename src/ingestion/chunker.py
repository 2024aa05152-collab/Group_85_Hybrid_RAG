import json
import re
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk text into overlapping segments without tiktoken dependency"""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size  # Approximate words
        self.chunk_overlap = chunk_overlap  # Approximate words
        self.word_chunk_size = chunk_size * 0.75  # Words per chunk (approx)
        self.word_overlap = chunk_overlap * 0.75  # Word overlap (approx)

    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())

    def count_tokens_simple(self, text: str) -> int:
        """Simple token estimation (words * 1.3)"""
        words = len(text.split())
        return int(words * 1.3)  # Rough estimate

    def split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        # Split by common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap using sentence boundaries"""
        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_id = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_word_count = len(sentence.split())

            # If adding this sentence would exceed chunk size and we have content
            if current_word_count + sentence_word_count > self.word_chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk_data(chunk_text, metadata, chunk_id, len(chunks)))
                chunk_id += 1

                # Start new chunk with overlap
                overlap_sentences = []
                overlap_word_count = 0

                # Go backwards to include overlap
                for j in range(len(current_chunk) - 1, -1, -1):
                    overlap_sentence = current_chunk[j]
                    overlap_sentence_words = len(overlap_sentence.split())

                    if overlap_word_count + overlap_sentence_words <= self.word_overlap:
                        overlap_sentences.insert(0, overlap_sentence)
                        overlap_word_count += overlap_sentence_words
                    else:
                        break

                current_chunk = overlap_sentences
                current_word_count = overlap_word_count
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
                i += 1

        # Add the last chunk if there's content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk_data(chunk_text, metadata, chunk_id, len(chunks)))

        # Add total_chunks to each chunk
        total = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total

        return chunks

    def _create_chunk_data(self, text: str, metadata: Dict[str, Any],
                           chunk_index: int, global_index: int) -> Dict[str, Any]:
        """Create chunk metadata dictionary"""
        return {
            "chunk_id": f"{metadata['url']}_{chunk_index}",
            "text": text,
            "token_count": self.count_tokens_simple(text),
            "word_count": len(text.split()),
            "url": metadata["url"],
            "title": metadata["title"],
            "chunk_index": chunk_index,
            "global_chunk_id": global_index,
            "total_chunks": None  # Will be filled later
        }

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all documents into chunks"""
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(doc["cleaned_text"], {
                "url": doc["url"],
                "title": doc["title"]
            })

            all_chunks.extend(chunks)

        # Update global chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk["global_chunk_id"] = i

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