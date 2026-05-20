# src/generation/llm_generator.py

from groq import Groq
from src.utils.helpers import load_config
import os

class LLMGenerator:

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.llm_config = self.config["llm"]
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = self.llm_config["model"]
        self.temperature = self.llm_config["temperature"]
        self.max_tokens = self.llm_config["max_tokens"]

    def generate(self, query: str, context: str) -> str:
        prompt = f"""You are a legal assistant answering questions about Indian law.

        Context from legal documents:
        {context}

        Question: {query}

        Answer based on the provided context and also explain a little based on the provided context. If the context doesn't contain relevant information, say "The provided context does not contain sufficient information."
        """

        message = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return message.choices[0].message.content

    def generate_with_metadata(self, query: str, context: str, retrieval_metadata: list) -> dict:
        answer = self.generate(query, context)

        return {
            "query": query,
            "answer": answer,
            "context_used": context,
            "retrieval_metadata": retrieval_metadata
        }
