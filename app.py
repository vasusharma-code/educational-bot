import sys
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError  # Import PdfReadError
from flask_cors import CORS  # Import CORS
from functools import wraps

# Add the parent directory of 'ai' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chat import chat_with_ai  # Import the AI chat function

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Configure the app for SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    text = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('files', lazy=True))

db.init_app(app)

# In-memory storage for users and responses
users = []
responses = {}

# In-memory database for storing file metadata
uploaded_files = []

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(filepath)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except PdfReadError:
        return None  # Return None if the PDF is invalid or corrupted

def authenticate_user(func):
    """Middleware to authenticate users based on user_id in the request body."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        data = request.get_json()
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required. Please provide user_id in the request body.'}), 401
        try:
            user_id = int(user_id)  # Ensure user_id is an integer
        except ValueError:
            return jsonify({'error': 'Invalid user_id format. Must be an integer.'}), 400
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'Invalid user_id. Authentication failed.'}), 401
        request.user = user  # Attach the authenticated user to the request object
        return func(*args, **kwargs)
    return wrapper

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if email is None or password is None:
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
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email, password=password).first()
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    return jsonify({'message': 'Login successful', 'user_id': user.id}), 200

@app.route('/api/chat', methods=['POST'])
@authenticate_user
def chat():
    data = request.get_json()
    message = data.get('message')

    # Fetch all uploaded files for the authenticated user
    user_files = UploadedFile.query.filter_by(user_id=request.user.id).all()
    user_data = [{'filename': file.filename, 'type': file.file_type, 'text': file.text} for file in user_files]

    # Use the AI chatbot to generate a response
    reply = chat_with_ai(message)

    # Save the response for the user (using IP as a simple identifier)
    user_ip = request.remote_addr
    if user_ip not in responses:
        responses[user_ip] = []
    responses[user_ip].append({'user': message, 'bot': reply})

    return jsonify({'reply': reply}), 200

@app.route('/api/responses', methods=['GET'])
def get_responses():
    user_ip = request.remote_addr
    user_responses = responses.get(user_ip, [])
    return jsonify(user_responses), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Extract user_id from form data
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'Authentication required. Please provide user_id in the form data.'}), 401
    try:
        user_id = int(user_id)  # Ensure user_id is an integer
    except ValueError:
        return jsonify({'error': 'Invalid user_id format. Must be an integer.'}), 400

    # Authenticate the user
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'Invalid user_id. Authentication failed.'}), 401

    # Handle file upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    file_type = request.form.get('type')

    if not file_type:
        return jsonify({'error': 'File type is required'}), 400

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(filepath)
        if pdf_text is None:
            return jsonify({'error': 'The uploaded file is not a valid PDF or is corrupted.'}), 400

        # Save metadata and text into the database
        uploaded_file = UploadedFile(filename=filename, file_type=file_type, text=pdf_text, user=user)
        db.session.add(uploaded_file)
        db.session.commit()

        return jsonify({'message': f'File {filename} uploaded successfully', 'type': file_type}), 200
    return jsonify({'error': 'Invalid file type. Only PDF files are allowed'}), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
