# src/generation_layer.py

import openai
from typing import List, Dict
import os
from openai import OpenAI

with open("secret_keys/openai_key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

# === Prompt Template ===

def build_prompt(query: str, chunks: List[Dict], few_shot: bool = False) -> str:
    prompt = "You are an assistant that summarizes and extracts insights from corporate email threads.\n"
    prompt += "Given a user query and the relevant email thread excerpts, answer the question concisely and accurately.\n\n"

    if few_shot:
        prompt += (
            "Example:\n"
            "Query: What was the decision on the marketing budget for Q2?\n"
            "Context:\n"
            "- The marketing team proposed a 20% increase for digital campaigns.\n"
            "- Finance approved a 10% increase after negotiation.\n"
            "Answer: A 10% increase in the Q2 marketing budget was approved after negotiation.\n\n"
        )

    prompt += f"Query: {query}\nContext:\n"
    for idx, chunk in enumerate(chunks):
        prompt += f"- {chunk['chunk']}\n"
    prompt += "\nAnswer:"
    return prompt


# === Generator Function ===

def generate_answer(query: str, chunks: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    prompt = build_prompt(query, chunks, few_shot=True)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Sorry, I couldn't generate a response due to an error."
