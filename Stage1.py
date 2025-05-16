import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

def load_metadata(file_path="metadata.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {i["pdf_url"]: i["summary"] for i in data}
def build_faiss_index(summaries, model):
    embeddings = np.array(model.encode(summaries)).astype('float32')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embeddings, index
def generate_synthetic_summary(llm, query):
    instructions = (
    "You are a highly knowledgeable assistant in natural language processing. "
    "Your task is to generate a synthetic research paper abstract that reformulates the following query with additional context on retrieving relevant research papers using Retrieval Augmented Generation (RAG). "
    "Ensure that the summary is informative, and clear."
    )
    complete_prompt = instructions + "\nQuery: " + query
    return llm(complete_prompt)
def main():
    dict_data = load_metadata()
    pdf_urls = list(dict_data.keys())
    summaries = list(dict_data.values())
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings, index = build_faiss_index(summaries, model)
    ollama_llm = OllamaLLM(model="deepseek-r1:7b")
    user_query = "I want to build a Bio medical NER model using Bert"
    if not user_query.strip():
        print("No query provided.")
        return
    synthetic_summary = generate_synthetic_summary(ollama_llm, user_query)
    print("\nSynthetic Summary:\n", synthetic_summary)
    query_embedding = model.encode([synthetic_summary]).astype('float32')
    _, top_indices = index.search(query_embedding, 10)
    print("\nTop 10 Relevant PDFs:")
    for idx in top_indices[0]:
        similarity_score = float(np.dot(embeddings[idx], query_embedding.T))
        print(f"URL: {pdf_urls[idx]} | Similarity: {similarity_score:.4f}")

if __name__ == "__main__":
    main()



#revelry or KIM
