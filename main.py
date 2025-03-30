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

# Initialize Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------ FAISS Vector Store ------------------------ #
def create_faiss_index():
    """Creates a FAISS vector store."""
    index = faiss.IndexFlatL2(384)  # 384 is the embedding size of MiniLM
    return index

def save_faiss_index(index, filepath):
    """Saves the FAISS index."""
    faiss.write_index(index, filepath)

def load_faiss_index(filepath):
    """Loads the FAISS index if it exists, else creates a new one."""
    return faiss.read_index(filepath) if os.path.exists(filepath) else create_faiss_index()

vector_store = load_faiss_index(VECTOR_STORE_PATH)
pdf_texts = []

# ------------------------ Text Extraction ------------------------ #
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def process_pdfs():
    """Processes PDFs, extracts text, and stores embeddings in FAISS."""
    global vector_store, pdf_texts

    vector_store.reset()  # Clear old embeddings
    pdf_texts = []  # Reset stored texts

    for file in os.listdir(UPLOAD_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(UPLOAD_DIR, file)
            text = extract_text_from_pdf(pdf_path)
            
            if text:
                paragraphs = [p.strip() for p in text.split("\n") if p.strip()]  # Remove empty lines

                for paragraph in paragraphs:
                    embedding = embedding_model.encode(paragraph, convert_to_numpy=True)
                    vector_store.add(np.array([embedding]))  # Store in FAISS
                    pdf_texts.append(paragraph)  # Keep track of text data

    save_faiss_index(vector_store, VECTOR_STORE_PATH)
    print(f"✅ Processed {len(pdf_texts)} text chunks from PDFs.")

# ------------------------ Extracting Major Topics ------------------------ #
def extract_subjects():
    """Extracts only major topics from syllabus."""
    major_topics = ["Computational Methods", "Discrete Mathematics", "Data Structures", "OOPs", "DLCS"]
    detected_topics = []

    all_text = " ".join(pdf_texts).lower()
    
    for topic in major_topics:
        if topic.lower() in all_text:
            detected_topics.append(topic)

    return detected_topics if detected_topics else ["No valid subjects found."]

# ------------------------ AI-Powered Notes Generation ------------------------ #
def retrieve_relevant_text(query):
    """Finds the most relevant text chunk from stored PDFs."""
    if not pdf_texts:
        return "⚠️ No PDFs processed yet."

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    distances, indices = vector_store.search(np.array([query_embedding]), k=1)

    # Check if any results were found
    if indices[0][0] == -1 or indices[0][0] >= len(pdf_texts):
        return "⚠️ No relevant content found in the syllabus."

    return pdf_texts[indices[0][0]]  # Retrieve closest matching text

def generate_notes(subject, unit):
    """Generates well-explained, detailed notes for students using AI."""
    relevant_text = retrieve_relevant_text(subject)  # Retrieve broader syllabus context

    prompts = [
        f"""You are a professor writing educational notes for the subject '{subject}' on the topic '{unit}'. 
        Create a **detailed explanation** with key concepts, examples, real-world applications, and summary.""",

        f"""Imagine you are explaining '{unit}' from '{subject}' to a beginner student. 
        Use easy-to-understand language and provide **structured notes** with examples and practical insights."""
    ]

    for attempt, prompt in enumerate(prompts):
        try:
            response = model.generate_content(prompt)
            if response and hasattr(response, "text"):
                return response.text
        except Exception as e:
            print(f"⚠️ AI Error: {e} (Attempt {attempt + 1})")

    return "⚠️ AI failed to generate notes. Try again with a different topic."

# ------------------------ Creating PDFs (Fixed Unicode Issue) ------------------------ #
class PDF(FPDF):
    """Custom PDF class to format the document properly."""
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(200, 10, "AI Generated Study Notes", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def clean_text(text):
    """Removes unprintable characters (fixes UnicodeEncodeError)."""
    return text.encode("ascii", "ignore").decode("ascii")

def create_pdf(content, output_file):
    """Creates a PDF with detailed, plain text notes (fixes Unicode issue)."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "", 12)

    paragraphs = content.split("\n")

    for paragraph in paragraphs:
        clean_paragraph = clean_text(paragraph)  # Remove unprintable Unicode characters
        pdf.multi_cell(0, 10, clean_paragraph)  # Normal text output

    pdf.output(output_file, "F")  # Ensure UTF-8 compatibility
    print(f"📄 PDF created: {output_file}")
    return output_file

# ------------------------ Main Chatbot ------------------------ #
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
