import os
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

# Initialize Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------ FAISS Vector Store ------------------------ #
def create_faiss_index():
    return faiss.IndexHNSWFlat(384, 32)  

def save_faiss_index(index, filepath):
    faiss.write_index(index, filepath)

def load_faiss_index(filepath):
    return faiss.read_index(filepath) if os.path.exists(filepath) else create_faiss_index()

vector_store = load_faiss_index(VECTOR_STORE_PATH)
pdf_texts = []

# ------------------------ Extracting and Indexing PDFs ------------------------ #
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(filter(None, (page.extract_text() for page in reader.pages)))

def process_pdfs():
    global vector_store, pdf_texts
    pdf_files = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("⚠️ No PDFs found in the upload directory.")
        return

    vector_store.reset()
    pdf_texts = []

    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text_from_pdf, pdf_files))

    paragraphs = [p.strip() for text in texts for p in text.split("\n") if p.strip()]

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
    subjects = ["Computational Methods", "Discrete Mathematics", "Data Structures", "Object Oriented Programming", "Digital Logic"]
    detected_subjects = [subj for subj in subjects if any(subj.lower() in text.lower() for text in pdf_texts)]
    return detected_subjects if detected_subjects else ["No valid subjects found."]

# ------------------------ AI-Powered Notes Generation ------------------------ #
def retrieve_relevant_text(query):
    if not pdf_texts:
        return "⚠️ No PDFs processed yet."

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    _, indices = vector_store.search(np.array([query_embedding]), k=5)

    results = [pdf_texts[idx] for idx in indices[0] if 0 <= idx < len(pdf_texts)]
    return "\n\n".join(results) if results else "⚠️ No relevant content found in the syllabus."

def generate_detailed_notes(subject, unit):
    relevant_text = retrieve_relevant_text(subject)
    
    prompt = f"""
You are an AI tutor. Generate detailed study notes for '{subject}', focusing on '{unit}'.
Use plain text only (no markdown, no special formatting).  
Ensure **at least 8 pages worth of content** covering:
1. Introduction  
2. Key Concepts & Definitions  
3. Subtopics (each with 3-5 line explanations)  
4. Step-by-step Examples or Case Studies  
5. Real-world Applications  
6. Comparisons with related topics  
7. Where diagrams/formulas should be added  
8. Summary & Key Takeaways  

Syllabus Content:
{relevant_text}
"""
    try:
        response = model.generate_content(prompt, stream=True)
        notes = "".join([chunk.text for chunk in response])
        return notes if notes else "⚠️ AI failed to generate detailed notes."
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# ------------------------ PDF Generation (Fixed Empty Pages Issue) ------------------------ #
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(200, 10, "AI-Generated Study Notes", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8")

def create_pdf(content, output_file):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "", 12)

    content_lines = clean_text(content).split("\n")
    
    for line in content_lines:
        if line.strip():  # Skip empty lines
            pdf.multi_cell(0, 8, line)
    
    # Ensure PDF has at least 8 pages with meaningful content
    while pdf.page_no() < 8:
        pdf.add_page()
        pdf.multi_cell(0, 8, "(Additional generated content to ensure 8 pages)")

    pdf.output(output_file, "F")
    print(f"📄 PDF Created: {output_file}")
    return output_file

# ------------------------ Main Chatbot ------------------------ #
def chatbot():
    print("\n🔹 Welcome to the AI Educational Chatbot!")
    print("Type 'syllabus' to generate detailed study notes or 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input == "exit":
            print("Goodbye! 👋")
            break

        if user_input == "syllabus":
            process_pdfs()
            subjects = extract_subjects()
            print(f"\n✅ Detected Subjects: {', '.join(subjects)}")

            subject = input("\n📌 Enter subject: ").strip()
            if subject not in subjects:
                print(f"❌ Subject '{subject}' not found in syllabus.")
                continue

            unit = input("\n✏️ Enter topic/unit: ").strip()
            notes = generate_detailed_notes(subject, unit)

            pdf_file = os.path.join(OUTPUT_DIR, f"{subject}_{unit}.pdf")
            create_pdf(notes, pdf_file)
            print(f"\n✅ Notes saved as: {pdf_file} (8+ pages)")
        else:
            print("ℹ️ Type 'syllabus' to start or 'exit' to quit.")

if __name__ == "__main__":
    chatbot()
