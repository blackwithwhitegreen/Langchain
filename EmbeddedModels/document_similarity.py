from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

document = [
    "The Godfather is often called the greatest film ever made",
    "Inception bends the rules of time and reality with stunning visuals",
    "Pulp Fiction redefined non-linear storytelling in Hollywood",
    "Interstellar explores love, time, and space through a father's sacrifice",
    "Parasite shows the harsh divide between social classes in South Korea",
    "The Dark Knight gave the world a legendary performance by Heath Ledger",
    "Forrest Gump reminds us that life is like a box of chocolates",
    "Avatar broke box office records with its revolutionary 3D effects",
    "The Shawshank Redemption is a timeless tale of hope and freedom",
    "Everything Everywhere All at Once blends chaos, comedy, and heart perfectly"

]

query  = "Tell me about Interstellar."

doc_embeddings = embedding.embed_documents(document)
query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embeddings],doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]

print(query)
print(document[index])
print("Similarlity Score is:",score)
