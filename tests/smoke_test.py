import os
from dotenv import load_dotenv
load_dotenv()

# 1. Node A — Gemini
from google import genai
gclient = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
gr = gclient.models.generate_content(
    model="gemini-3.5-flash",
    contents="Say OK",
)
print("Gemini:", gr.text.strip())

# 2. Node B — OpenAI
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
r = client.responses.create(
    model="gpt-5.6-luna",
    input="Say OK",
)
print("OpenAI:", r.output_text.strip())

# 3. Local embedding model
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding dim:", len(enc.encode("test")))

# 4. ChromaDB
import chromadb
c = chromadb.PersistentClient(path="./chroma_db")
col = c.get_or_create_collection("smoke")
col.add(ids=["1"], documents=["The capital of France is Paris."])
print("Chroma query:", col.query(query_texts=["France capital"], n_results=1))

# 5. Ragas imports
import ragas
print("Ragas OK:", ragas.__version__)

# 6. Stats
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
print("Stats OK")

print("\n--- ALL SYSTEMS GO ---")