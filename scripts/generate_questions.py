#!/usr/bin/env python3
"""
Generate evaluation questions from Wikipedia corpus
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import logging
import re
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FILE_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generate diverse questions from Wikipedia chunks"""

    def __init__(self):
        self.question_templates = {
            "factual": [
                "What is {entity}?",
                "When was {entity} {action}?",
                "Who {action} {entity}?",
                "Where is {entity} located?",
                "How does {entity} work?",
                "What are the main characteristics of {entity}?",
                "Why is {entity} important?",
                "What is the purpose of {entity}?"
            ],
            "comparative": [
                "What are the differences between {entity1} and {entity2}?",
                "How is {entity1} similar to {entity2}?",
                "Compare {entity1} with {entity2}.",
                "What advantages does {entity1} have over {entity2}?",
                "How does {entity1} differ from {entity2} in terms of {aspect}?"
            ],
            "inferential": [
                "Based on the information, what can be inferred about {entity}?",
                "What are the implications of {fact}?",
                "If {condition}, what would happen to {entity}?",
                "What conclusions can be drawn about {entity}?",
                "How might {entity} affect {other_entity}?"
            ],
            "multi-hop": [
                "What is the relationship between {entity1} and {entity2} through {entity3}?",
                "How did {event1} lead to {event2}?",
                "What are the common factors between {entity1} and {entity2} in the context of {topic}?",
                "Based on information about {entity1} and {entity2}, what can be said about {entity3}?"
            ]
        }

        # Actions for factual questions
        self.actions = ["created", "invented", "discovered", "founded", "developed", "built", "established"]

        # Aspects for comparative questions
        self.aspects = ["performance", "efficiency", "cost", "complexity", "applications", "history"]

        # Conditions for inferential questions
        self.conditions = ["this trend continues", "the technology improves", "more funding is available",
                           "regulations change"]

    def extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text"""
        entities = []

        # Look for capitalized phrases (simple NER)
        sentences = text.split('. ')

        for sentence in sentences:
            # Find noun phrases with initial capitals
            words = sentence.split()
            for i in range(len(words) - 1):
                if (words[i][0].isupper() and words[i + 1][0].isupper() and
                        len(words[i]) > 1 and len(words[i + 1]) > 1):
                    entity = f"{words[i]} {words[i + 1]}"
                    if entity not in entities and len(entity) > 3:
                        entities.append(entity)
                elif words[i][0].isupper() and len(words[i]) > 2 and words[i].isalpha():
                    if words[i] not in entities:
                        entities.append(words[i])

        return list(set(entities))[:10]

    def extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text"""
        facts = []
        sentences = text.split('. ')

        for sentence in sentences:
            # Look for factual patterns
            if any(pattern in sentence.lower() for pattern in
                   ['is a', 'was founded', 'located in', 'developed by',
                    'created in', 'invented by', 'born in', 'died in',
                    'consists of', 'includes', 'contains', 'provides']):
                facts.append(sentence.strip())

        return facts[:5]

    def generate_question_from_chunk(self, chunk: Dict[str, Any],
                                     question_type: str) -> Dict[str, Any]:
        """Generate a question from a single chunk"""
        text = chunk["text"]
        entities = self.extract_entities(text)
        facts = self.extract_facts(text)

        if not entities:
            return None

        # Select template
        templates = self.question_templates.get(question_type, [])
        if not templates:
            return None

        template = random.choice(templates)

        # Fill template
        question = template
        if "{entity}" in template:
            entity = random.choice(entities)
            question = question.replace("{entity}", entity)

            # Add action for some templates
            if "{action}" in question:
                action = random.choice(self.actions)
                question = question.replace("{action}", action)

        elif "{entity1}" in template and "{entity2}" in template:
            if len(entities) >= 2:
                entity1, entity2 = random.sample(entities, 2)
                question = question.replace("{entity1}", entity1)
                question = question.replace("{entity2}", entity2)

                # Add aspect for some templates
                if "{aspect}" in question:
                    aspect = random.choice(self.aspects)
                    question = question.replace("{aspect}", aspect)
            else:
                return None
        elif "{fact}" in template:
            if facts:
                fact = random.choice(facts)
                question = question.replace("{fact}", fact[:50] + "...")
            else:
                return None
        elif "{condition}" in template:
            condition = random.choice(self.conditions)
            question = question.replace("{condition}", condition)
            if "{entity}" in question:
                entity = random.choice(entities)
                question = question.replace("{entity}", entity)
        elif "{topic}" in template:
            # Use first entity as topic
            topic = entities[0] if entities else "technology"
            question = question.replace("{topic}", topic)

        # Generate answer (simplified)
        sentences = text.split('. ')
        answer = sentences[0] if sentences else "Information not found."

        return {
            "id": f"q_{hash(chunk['chunk_id']) % 1000000:06d}",
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "source_chunk_id": chunk["chunk_id"],
            "source_url": chunk["url"],
            "source_title": chunk["title"],
            "source_text_preview": text[:100] + "...",
            "source_urls": [chunk["url"]]
        }

    def generate_questions_from_chunks(self, chunks: List[Dict[str, Any]],
                                       num_questions: int = 100) -> List[Dict[str, Any]]:
        """Generate questions from chunks"""
        questions = []

        # Filter chunks with enough text
        valid_chunks = [c for c in chunks if len(c["text"].split()) > 30]

        if not valid_chunks:
            logger.error("No valid chunks found")
            return questions

        # Shuffle chunks
        random_chunks = random.sample(valid_chunks, min(len(valid_chunks), 200))

        # Balance question types
        question_types = list(self.question_templates.keys())
        questions_per_type = num_questions // len(question_types)

        logger.info(f"Generating questions from {len(random_chunks)} chunks...")

        for q_type in question_types:
            type_questions = []
            attempts = 0
            max_attempts = questions_per_type * 10

            while len(type_questions) < questions_per_type and attempts < max_attempts:
                chunk = random.choice(random_chunks)
                question = self.generate_question_from_chunk(chunk, q_type)

                if question and question not in type_questions:
                    type_questions.append(question)

                attempts += 1

            questions.extend(type_questions)
            logger.info(f"Generated {len(type_questions)} {q_type} questions")

        # Ensure we have exactly num_questions
        if len(questions) > num_questions:
            questions = questions[:num_questions]

        logger.info(f"Generated {len(questions)} total questions")
        return questions

    def save_questions(self, questions: List[Dict[str, Any]], output_path: Path):
        """Save questions to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(questions)} questions to {output_path}")


def main():
    """Main function for standalone question generation"""
    # Load chunks
    chunks_path = FILE_PATHS["corpus_chunks"]

    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        logger.error("Please run ingestion first or provide path to existing chunks.")
        return

    # Load chunks
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # Generate questions
    generator = QuestionGenerator()
    questions = generator.generate_questions_from_chunks(chunks, num_questions=100)

    # Save questions
    output_path = FILE_PATHS["questions"]
    generator.save_questions(questions, output_path)

    # Print sample questions
    print("\n" + "=" * 60)
    print("Sample Generated Questions:")
    print("=" * 60)
    for i, q in enumerate(questions[:5]):
        print(f"\n{i + 1}. Type: {q['question_type']}")
        print(f"   Question: {q['question']}")
        print(f"   Answer: {q['answer'][:80]}...")
        print(f"   Source: {q['source_title']}")


if __name__ == "__main__":
    main()