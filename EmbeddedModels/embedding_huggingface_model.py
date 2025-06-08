from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# text = "Delhi is the captial of India"

document = [
    "Delhi is the Capital of India",
    "Virat is the Cricketer",
    "IIOT is the Computer Science Branch.",
    "Run run run"
]

vector = embedding.embed_documents(document)

print(str(vector))