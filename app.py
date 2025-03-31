import os
import sys
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from fpdf import FPDF
from functools import wraps

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chat import chat_with_ai  # Import AI chatbot function

app = Flask(__name__)
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Directory Setup
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data/uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'data/outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------- Database Models -------------------------- #
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False, default="syllabus")
    text = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('files', lazy=True))

class Embedding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('uploaded_file.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text_chunk = db.Column(db.Text, nullable=False)
    vector = db.Column(db.PickleType, nullable=False)

    file = db.relationship('UploadedFile', backref=db.backref('embeddings', lazy=True))
    user = db.relationship('User', backref=db.backref('embeddings', lazy=True))

# -------------------------- Helper Functions -------------------------- #
def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(filepath)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except PdfReadError:
        return None

def process_uploaded_pdf(filepath, user_id, file_id):
    """Extracts text, computes embeddings, and stores them in the database."""
    text = extract_text_from_pdf(filepath)
    if not text:
        return None

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    embeddings = embedding_model.encode(paragraphs, convert_to_numpy=True)

    for i, paragraph in enumerate(paragraphs):
        embedding_entry = Embedding(file_id=file_id, user_id=user_id, text_chunk=paragraph, vector=embeddings[i].tolist())
        db.session.add(embedding_entry)
    db.session.commit()

    return True

class PDF(FPDF):
    """Custom PDF class with header and footer."""
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(200, 10, "AI Generated Study Notes", ln=True, align="C")
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def create_pdf(content, output_file):
    """Creates a PDF from the given text."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, content)
    pdf.output(output_file)
    return output_file

# -------------------------- Middleware -------------------------- #
def authenticate_user(func):
    """Middleware to authenticate users."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        data = request.get_json()
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required.'}), 401
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'Invalid user_id.'}), 401
        request.user = user
        return func(*args, **kwargs)
    return wrapper

# -------------------------- API Routes -------------------------- #
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email, password = data.get('email'), data.get('password')
    if not email or not password:
        return jsonify({'message': 'Enter email and password'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'User already exists'}), 400
    user = User(email=email, password=password)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'User created'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email, password = data.get('email'), data.get('password')
    user = User.query.filter_by(email=email, password=password).first()
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    return jsonify({'message': 'Login successful', 'user_id': user.id}), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    user_id = request.form.get('user_id')
    if not user_id or not User.query.get(user_id):
        return jsonify({'error': 'Invalid user_id'}), 401

    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    if text is None:
        return jsonify({'error': 'Invalid PDF file'}), 400

    uploaded_file = UploadedFile(filename=filename, file_type="syllabus", text=text, user_id=user_id)
    db.session.add(uploaded_file)
    db.session.commit()

    process_uploaded_pdf(filepath, user_id, uploaded_file.id)
    return jsonify({'message': 'File uploaded and processed'}), 200

@app.route('/api/chat', methods=['POST'])
@authenticate_user
def chat():
    message = request.get_json().get('message')
    query_embedding = embedding_model.encode([message])[0]

    embeddings = Embedding.query.filter_by(user_id=request.user.id).all()
    
    if embeddings:
        vectors = np.array([e.vector for e in embeddings])
        texts = [e.text_chunk for e in embeddings]

        distances = cdist([query_embedding], vectors, metric="cosine")[0]
        top_chunks = "\n\n".join([texts[i] for i in np.argsort(distances)[:5]])

        response = chat_with_ai(f"Context:\n{top_chunks}\n\nQuestion: {message}")
    else:
        response = chat_with_ai(message)  # Normal chatbot mode if no PDFs

    return jsonify({'reply': response}), 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
