from flask import Flask, render_template, request, jsonify
from chatbot import chat

# Initialize Flask app
app = Flask(__name__)

# Render the index.html template.
@app.route('/')
def index():
    return render_template('index.html')

# Handle requests to '/chat' endpoint for text-based communication.
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "No message received"}), 400
    response = chat(user_input)
    return jsonify({"response": response})

# Handle requests to '/voice' endpoint for speech-to-text communication.
@app.route('/voice', methods=['POST'])
def speech_to_text():
    if 'text_data' not in request.form:
        return jsonify({'success': False, 'error': 'No text data found'}), 400

    text = request.form['text_data']
    response = chat(text)
    return jsonify({'success': True, 'response': response})

if __name__ == '__main__':
    app.run(debug=True)
