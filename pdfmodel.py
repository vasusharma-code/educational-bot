import os
import re
import json
import faiss
import numpy as np
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fpdf import FPDF
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
OUTPUT_DIR = "data/outputs"
VECTOR_STORE_PATH = "data/vector_store.index"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Embedding Model (Batch Processing for Speed)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------ FAISS Vector Store (Optimized) ------------------------ #
def create_faiss_index():
    """Creates a FAISS HNSW vector store for faster retrieval."""
    index = faiss.IndexHNSWFlat(384, 32)  # HNSW for 10-100x speedup
    return index

def save_faiss_index(index, filepath):
    """Saves the FAISS index."""
    faiss.write_index(index, filepath)

def load_faiss_index(filepath):
    """Loads the FAISS index if it exists, else creates a new one."""
    return faiss.read_index(filepath) if os.path.exists(filepath) else create_faiss_index()

vector_store = load_faiss_index(VECTOR_STORE_PATH)
pdf_texts = []

# ------------------------ Fast Text Extraction (Parallelized) ------------------------ #
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    return "\n".join(filter(None, (page.extract_text() for page in reader.pages)))

def process_pdfs():
    """Processes PDFs in parallel, extracts text, and stores embeddings efficiently."""
    global vector_store, pdf_texts

    pdf_files = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No PDFs found in the upload directory.")
        return

    vector_store.reset()  # Clear old embeddings
    pdf_texts = []  # Reset stored texts

    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text_from_pdf, pdf_files))

    paragraphs = [p.strip() for text in texts for p in text.split("\n") if p.strip()]
    
    # Encode in **batches** for 10x speed improvement
    batch_size = 32
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i+batch_size]
        embeddings = embedding_model.encode(batch, convert_to_numpy=True)
        vector_store.add(embeddings)
        pdf_texts.extend(batch)

    save_faiss_index(vector_store, VECTOR_STORE_PATH)
    print(f"✅ Processed {len(pdf_texts)} text chunks from PDFs.")

# ------------------------ Extracting Major Topics ------------------------ #
def extract_subjects():
    """Extracts only major topics from syllabus."""
    major_topics = ["Computational Methods", "Discrete Mathematics", "Data Structures", "OOPs", "DLCS"]
    all_text = " ".join(pdf_texts).lower()
    return [topic for topic in major_topics if topic.lower() in all_text] or ["No valid subjects found."]

# ------------------------ AI-Powered Notes Generation ------------------------ #
def retrieve_relevant_text(query):
    """Finds the most relevant text chunk from stored PDFs."""
    if not pdf_texts:
        return "⚠️ No PDFs processed yet."

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    _, indices = vector_store.search(np.array([query_embedding]), k=3)  # Retrieve top-3 matches

    results = [pdf_texts[idx] for idx in indices[0] if idx >= 0 and idx < len(pdf_texts)]
    return "\n\n".join(results) if results else "⚠️ No relevant content found in the syllabus."

def generate_notes(subject, unit):
    """Generates concise, structured notes for students."""
    relevant_text = retrieve_relevant_text(subject)

    prompt = f"""
You are an AI tutor. Generate **short, structured, and high-quality study notes** for '{subject}', focusing on '{unit}'.
Include:
- **Introduction**: Briefly explain "{unit}".
- **Key Concepts**: Bullet points with short explanations.
- **Examples**: Practical applications.
- **Real-world Applications**: Where it's used.
- **Summary**: Main takeaways.

Relevant syllabus content:
{relevant_text}
"""

    try:
        response = model.generate_content(prompt, stream=True)
        notes = "".join([chunk.text for chunk in response])
        return notes if notes else "⚠️ AI failed to generate notes."
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# ------------------------ Creating PDFs (Optimized) ------------------------ #
class PDF(FPDF):
    """Custom PDF class with headers/footers."""
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(200, 10, "AI Generated Study Notes", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def clean_text(text):
    """Removes unprintable characters."""
    return text.encode("ascii", "ignore").decode("ascii")

def create_pdf(content, output_file):
    """Creates a structured PDF with notes."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "", 12)

    for paragraph in content.split("\n"):
        pdf.multi_cell(0, 8, clean_text(paragraph))

    pdf.output(output_file)
    print(f"📄 PDF created: {output_file}")
    return output_file

# ------------------------ Main Chatbot (Optimized) ------------------------ #
def chatbot():
    """Interactive chatbot for syllabus-based note generation."""
    print("\n🔹 Welcome to the AI Educational Chatbot!")
    print("Type 'syllabus' to generate notes or 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input == "exit":
            print("Goodbye! 👋")
            break

        if user_input == "syllabus":
            process_pdfs()
            subjects = extract_subjects()
            print(f"\n✅ Detected major subjects: {', '.join(subjects)}")
            
            subject = input("\n📌 Enter subject: ").strip()
            if subject not in subjects:
                print(f"❌ Subject '{subject}' not found in syllabus.")
                continue

            unit = input("\n✏️ Enter topic/unit: ").strip()
            notes = generate_notes(subject, unit)
            
            pdf_file = os.path.join(OUTPUT_DIR, f"{subject}_{unit}.pdf")
            create_pdf(notes, pdf_file)
            print(f"\n✅ Notes saved as: {pdf_file}")
        else:
            print("ℹ️ Type 'syllabus' to start or 'exit' to quit.")

if __name__ == "__main__":
    chatbot()