# Use a pipeline as a high-level helper
from transformers import pipeline

# Load the model
pipe = pipeline("text-generation", model="sarvamai/sarvam-m")

# Format the message like a chat history
messages = [
    {"role": "user", "content": "Who are you?"}
]

# Convert to a single prompt string
prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

# Generate text
output = pipe(prompt, max_new_tokens=100, do_sample=True)

# Show the result
print(output[0]['generated_text'])
