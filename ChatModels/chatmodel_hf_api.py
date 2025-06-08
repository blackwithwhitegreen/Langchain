from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="sarvamai/sarvam-m",
    task="text-generation"
)

result = llm.invoke("What is the capital of India?")
print(result)
