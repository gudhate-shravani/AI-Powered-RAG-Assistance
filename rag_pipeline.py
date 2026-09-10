import os

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# ENVIRONMENT & GROQ
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found in .env file."
    )

client = Groq(api_key=api_key)


# ============================================================
# EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=50
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_document(pdf_file):

    text = extract_text_from_pdf(
        pdf_file
    )

    if not text.strip():
        raise ValueError(
            "Could not extract text from this PDF."
        )

    chunks = create_chunks(text)

    index = create_faiss_index(chunks)

    return chunks, index


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    query,
    index,
    chunks,
    top_k=4
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for position, i in enumerate(indices[0]):

        if i != -1:

            results.append(
                {
                    "chunk": chunks[i],
                    "distance": float(
                        distances[0][position]
                    ),
                    "index": int(i)
                }
            )

    return results


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    query,
    retrieved_chunks
):

    context = "\n\n".join(
        item["chunk"]
        for item in retrieved_chunks
    )

    prompt = f"""
You are a reliable document-based AI assistant.

Answer the user's question using ONLY the
provided context.

Rules:
1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   say:
   "I could not find this information in the
   provided document."
4. Keep the answer clear and useful.
5. Use bullet points when appropriate.

Context:
--------------------------------------------------
{context}
--------------------------------------------------

Question:
{query}
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a reliable RAG "
                    "document assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content


# ============================================================
# COMPLETE RAG QUERY
# ============================================================

def ask_question(
    query,
    index,
    chunks,
    top_k=4
):

    retrieved_chunks = retrieve_chunks(
        query,
        index,
        chunks,
        top_k
    )

    answer = generate_answer(
        query,
        retrieved_chunks
    )

    return answer, retrieved_chunks


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    pdf_path = "data/documents/sample.pdf"

    print("\n==============================")
    print("       RAG ASSISTANT")
    print("==============================")

    print("\nProcessing PDF...")

    with open(pdf_path, "rb") as pdf:

        chunks, index = process_document(pdf)

    print(
        "Total chunks:",
        len(chunks)
    )

    print(
        "Total vectors:",
        index.ntotal
    )

    query = input(
        "\nAsk a question: "
    )

    answer, sources = ask_question(
        query,
        index,
        chunks
    )

    print("\n==============================")
    print("AI ANSWER")
    print("==============================")

    print(answer)