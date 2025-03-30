import sys
import os
from flask import Flask, request, jsonify

# Add the parent directory of 'ai' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chat import chat_with_ai  # Import the AI chat function

app = Flask(__name__)

# In-memory storage for users and responses
users = []
responses = {}

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if email is None or password is None:
        return jsonify({'message': 'enter email and password'})

    if any(user['email'] == email for user in users):
        return jsonify({'error': 'User already exists'}), 400

    users.append({'email': email, 'password': password})
    return jsonify({'message': 'User created'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = next((user for user in users if user['email'] == email and user['password'] == password), None)
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    return jsonify({'message': 'Login successful'}), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
