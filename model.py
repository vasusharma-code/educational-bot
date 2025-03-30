import os
import re
import json
import requests
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fpdf import FPDF

# Load API key
load_dotenv()
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
LLAMA_API_URL = "https://api.llama-api.com/v1/generate"

# Directory Setup
UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------ Text Extraction ------------------------ #
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def extract_subjects(syllabus_text):
    """Extracts high-level subjects from the syllabus."""
    pattern = r"\b(?:Mathematics|Physics|Chemistry|Computer Science|Data Structures|Algorithms|Machine Learning|Artificial Intelligence|Cyber Security|Engineering Mechanics|Graph Theory|Operating Systems)\b"
    subjects = list(set(re.findall(pattern, syllabus_text, re.IGNORECASE)))
    return subjects if subjects else ["No valid subjects found."]

def process_syllabus():
    """Finds the syllabus PDF, extracts text, and detects subjects."""
    syllabus_files = [f for f in os.listdir(UPLOAD_DIR) if "syllabus" in f.lower() and f.lower().endswith(".pdf")]
    if not syllabus_files:
        return None, None
    syllabus_path = os.path.join(UPLOAD_DIR, syllabus_files[0])
    syllabus_text = extract_text_from_pdf(syllabus_path)
    subjects = extract_subjects(syllabus_text)
    return syllabus_text, subjects

# ------------------------ AI Processing (LLama API) ------------------------ #
def call_llama_api(prompt):
    """Calls LLama API with the given prompt."""
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}", "Content-Type": "application/json"}
    data = {"prompt": prompt, "max_tokens": 500}
    
    response = requests.post(LLAMA_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        print(f"❌ LLama API Error: {response.text}")
        return ""

def analyze_units(subject, syllabus_text):
    """Extracts topics/units from a subject using LLama API."""
    prompt = f"""
You are an expert academic assistant. Extract key topics or units for the subject "{subject}" based on the syllabus.
Return them as a **clean JSON list of strings**. Example output: ["Unit 1", "Unit 2", "Unit 3"]
Syllabus:
{syllabus_text}
"""
    response_text = call_llama_api(prompt)
    
    try:
        units = json.loads(response_text)
        if isinstance(units, list) and all(isinstance(unit, str) for unit in units):
            return units
    except json.JSONDecodeError:
        print("⚠️ AI returned an invalid response. Retrying with a simpler query...")
    
    # Retry with a simpler prompt
    retry_prompt = f"List the main units/topics for '{subject}' as a JSON array."
    response_text = call_llama_api(retry_prompt)
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print("❌ Failed to extract units.")
        return []

def generate_notes(subject, syllabus_text, unit):
    """Generates detailed notes for a given subject and unit using LLama API."""
    prompt = f"""
You are an AI tutor. Generate well-structured notes for "{unit}" in "{subject}" based on the syllabus.
Include:
1. **Introduction**
2. **Key Concepts**
3. **Formulas & Examples**
4. **Practical Applications**
5. **Summary**
Ensure the response is structured and formatted well.
Syllabus:
{syllabus_text}
"""
    return call_llama_api(prompt)

# ------------------------ PDF Generation ------------------------ #
def create_pdf(content, output_file):
    """Creates a PDF from extracted notes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in content.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_file)
    print(f"📄 PDF created: {output_file}")
    return output_file

# ------------------------ Chatbot ------------------------ #
def chatbot():
    """Main chatbot loop."""
    print("\n🔹 Welcome to the AI Educational Chatbot!")
    print("Type 'syllabus' to generate notes or 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input == "exit":
            print("Goodbye! 👋")
            break

        if user_input == "syllabus":
            syllabus_text, subjects = process_syllabus()
            if not syllabus_text:
                print("⚠️ No syllabus file found. Please upload one in the 'data/uploads' folder.")
                continue
            
            print(f"\n✅ Detected subjects: {', '.join(subjects)}")
            subject = input("\n📌 Enter subject: ").strip()
            if subject not in subjects:
                print(f"❌ Subject '{subject}' not found in syllabus.")
                continue
            
            units = analyze_units(subject, syllabus_text)
            if units:
                print(f"\n📚 Topics detected for {subject}:")
                for idx, unit in enumerate(units, start=1):
                    print(f"{idx}. {unit}")

                unit = input("\n✏️ Enter unit number or name: ").strip()
                try:
                    unit_idx = int(unit)
                    if 1 <= unit_idx <= len(units):
                        selected_unit = units[unit_idx - 1]
                    else:
                        print("❌ Invalid unit number.")
                        continue
                except ValueError:
                    if unit in units:
                        selected_unit = unit
                    else:
                        print("❌ Invalid unit name.")
                        continue

                notes = generate_notes(subject, syllabus_text, selected_unit)
                pdf_file = os.path.join(OUTPUT_DIR, f"{subject}_{selected_unit}.pdf")
                create_pdf(notes, pdf_file)
                print(f"\n✅ Notes saved as: {pdf_file}")
            else:
                print(f"❌ No units found for {subject}.")
        else:
            print("ℹ️ Type 'syllabus' to start or 'exit' to quit.")

if __name__ == "__main__":
    chatbot()
