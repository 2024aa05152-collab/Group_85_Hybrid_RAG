import logging
from typing import List, Dict, Any
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMGenerator:
    """Generate answers using various LLM backends"""

    def __init__(self, use_local=False, model_name="gpt-3.5-turbo"):
        self.use_local = use_local
        self.model_name = model_name
        self.api_key = None

    def create_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Create prompt for LLM"""
        # Combine context chunks
        context = "\n\n".join([
            f"Source: {chunk['title']}\nContent: {chunk['text'][:500]}..."
            for chunk in context_chunks[:3]  # Limit to 3 chunks
        ])

        # Create prompt
        prompt = f"""Based ONLY on the following information, answer the question accurately and concisely.

Context Information:
{context}

Question: {query}

Important Instructions:
1. Answer based ONLY on the provided context
2. If the information is not in the context, say "I cannot answer based on the provided information"
3. Keep the answer concise and to the point
4. Include relevant details from the context

Answer: """

        return prompt

    def generate_with_openai(self, prompt: str) -> str:
        """Generate answer using OpenAI API"""
        try:
            import openai
            # Configure OpenAI
            openai.api_key = self.api_key or "your-api-key-here"  # Replace with your key

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error generating answer: {str(e)[:100]}"

    def generate_with_huggingface(self, prompt: str) -> str:
        """Generate answer using HuggingFace API"""
        try:
            # You need a HuggingFace API token
            API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
            headers = {"Authorization": "Bearer hf_your_token_here"}  # Replace with your token

            response = requests.post(API_URL, headers=headers, json={
                "inputs": prompt,
                "parameters": {
                    "max_length": 200,
                    "temperature": 0.7,
                    "top_p": 0.95
                }
            })

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', 'No response')
            return "Error: Failed to get response"

        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            return f"Error: {str(e)[:100]}"

    def generate_simple_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate a simple answer by extracting relevant text"""
        # Simple extraction-based answer
        query_terms = query.lower().split()
        relevant_texts = []

        for chunk in context_chunks:
            text = chunk["text"].lower()
            # Count matching terms
            matches = sum(1 for term in query_terms if term in text and len(term) > 3)
            if matches > 0:
                # Extract sentence containing query terms
                sentences = chunk["text"].split('. ')
                for sentence in sentences:
                    if any(term in sentence.lower() for term in query_terms if len(term) > 3):
                        relevant_texts.append(sentence)
                        break

        if relevant_texts:
            # Combine and summarize
            answer = '. '.join(relevant_texts[:3]) + '.'
            return f"Based on the information: {answer}"
        else:
            return "I cannot find specific information to answer this question based on the provided context."

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer based on query and context"""
        try:
            prompt = self.create_prompt(query, context_chunks)

            if self.use_local:
                # Try local model first
                try:
                    answer = self.generate_with_local_model(prompt)
                except:
                    answer = self.generate_simple_answer(query, context_chunks)
            else:
                # Try external APIs
                try:
                    answer = self.generate_with_openai(prompt)
                except:
                    try:
                        answer = self.generate_with_huggingface(prompt)
                    except:
                        answer = self.generate_simple_answer(query, context_chunks)

            # Prepare response
            response = {
                "query": query,
                "answer": answer,
                "context_chunks": context_chunks,
                "num_chunks_used": len(context_chunks),
                "success": True
            }

            logger.info(f"Generated answer (first 100 chars): {answer[:100]}...")
            return response

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "query": query,
                "answer": f"Error: {str(e)}",
                "context_chunks": context_chunks,
                "success": False,
                "error": str(e)
            }