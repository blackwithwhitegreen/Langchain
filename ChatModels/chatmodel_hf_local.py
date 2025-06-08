from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load the tokenizer and model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_id = "microsoft/DialoGPT-medium"
# model_id = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Create a Transformers pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1
)

# Integrate with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Ask a question
question = "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is the reason UFC is more famous in the West Countries?\n<|assistant|>\n"
# question = "Who is the best Cricket Player?"

response = llm.invoke(question)
print("Answer:", response)
