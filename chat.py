import os
import faiss
import numpy as np
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Google Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Directory Setup
UPLOAD_DIR = "data/uploads"
VECTOR_STORE_PATH = "data/vector_store.index"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------ FAISS Optimized Vector Store ------------------------ #
def create_faiss_index():
    """Creates a FAISS vector store using Inner Product for faster search."""
    index = faiss.IndexFlatIP(384)  # Inner Product for faster similarity search
    return index

def save_faiss_index(index, filepath):
    """Saves the FAISS index."""
    faiss.write_index(index, filepath)

def load_faiss_index(filepath):
    """Loads or initializes the FAISS index."""
    return faiss.read_index(filepath) if os.path.exists(filepath) else create_faiss_index()

vector_store = load_faiss_index(VECTOR_STORE_PATH)
pdf_texts = []

# ------------------------ Fast Text Extraction ------------------------ #
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file in parallel."""
    reader = PdfReader(pdf_path)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

def process_pdfs():
    """Processes PDFs, extracts text, and stores embeddings in FAISS."""
    global vector_store, pdf_texts

    vector_store.reset()  # Clear old embeddings
    pdf_texts = []  # Reset stored texts
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]

    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text_from_pdf, [os.path.join(UPLOAD_DIR, f) for f in files]))

    paragraphs = [p.strip() for text in texts for p in text.split("\n") if p.strip()]
    
    # Encode in batches for speed
    embeddings = embedding_model.encode(paragraphs, convert_to_numpy=True, batch_size=16)
    
    # Normalize embeddings for FAISS IP search
    faiss.normalize_L2(embeddings)
    vector_store.add(embeddings)
    pdf_texts.extend(paragraphs)

    save_faiss_index(vector_store, VECTOR_STORE_PATH)
    print(f"✅ Processed {len(pdf_texts)} text chunks from PDFs.")

# ------------------------ Fast AI-Powered Q&A ------------------------ #
def retrieve_relevant_text(query):
    """Finds the most relevant text chunk from stored PDFs."""
    if not pdf_texts:
        return "⚠️ No PDFs processed yet."

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    faiss.normalize_L2(query_embedding.reshape(1, -1))  # Normalize for IP search

    distances, indices = vector_store.search(np.array([query_embedding]), k=3)

    # Return top 3 relevant chunks
    results = [pdf_texts[idx] for idx in indices[0] if idx >= 0 and idx < len(pdf_texts)]
    return "\n\n".join(results) if results else "⚠️ No relevant content found."

def chat_with_ai(user_query):
    """AI-enhanced Q&A system for quick topic explanations."""
    context = retrieve_relevant_text(user_query)

    prompt = f"""
    You are an expert professor answering a student's question: "{user_query}".
    Based on the syllabus context provided below, give a **concise, structured, and easy-to-understand** answer.
    Context: {context}
    """

    try:
        response = model.generate_content(prompt)
        return response.text if response else "⚠️ AI couldn't generate a response."
    except Exception as e:
        print(f"⚠️ AI Error: {e}")
        return "⚠️ Something went wrong. Try again."

# ------------------------ Interactive Chatbot ------------------------ #
def chatbot():
    """AI chatbot that answers questions about topics directly."""
    print("\n🔹 Welcome to the AI Educational Chatbot!")
    print("Ask any topic-related question or type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break

        response = chat_with_ai(user_input)
        print(f"\n🤖 AI: {response}")

if __name__ == "__main__":
    process_pdfs()  # Ensure PDFs are processed before chatting
    chatbot()
