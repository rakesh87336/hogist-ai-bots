# hogist_ai.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Detect device: GPU if available else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load FLAN-T5 Small model and tokenizer
LOCAL_MODEL_PATH = "local_flan_t5_small"
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH)

# Move model to device
model.to(device)

# Create text2text generation pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    device=0 if torch.cuda.is_available() else -1
)

# Intent-to-route mapping and knowledge base remain unchanged
INTENT_MAP = {
    # ... same as before ...
}

KNOWLEDGE_RESPONSES = {
    # ... same as before ...
}

@app.route("/")
def index():
    return "Hogist Voice Bot Server Running"

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('user_message')
def handle_user_message(data):
    user_query = data.get("query", "").strip().lower()
    print(f"User said: {user_query}")

    # Check for intent match
    matched_intent = None
    for key in INTENT_MAP:
        if key in user_query:
            matched_intent = INTENT_MAP[key]
            break

    if matched_intent:
        prompt = matched_intent["prompt"]
        clean_response = generate_response(prompt)
        emit("bot_response", {"response": clean_response, "route": matched_intent["route"]})
        return

    # Check for knowledge-based question
    for key in KNOWLEDGE_RESPONSES:
        if key in user_query:
            response_text = KNOWLEDGE_RESPONSES[key].strip()
            emit("bot_response", {"response": response_text})
            return

    # Fallback general query
    clean_response = generate_response(user_query) + " Still not sure what you mean? Ask me anything!"
    emit("bot_response", {"response": clean_response})

def generate_response(prompt):
    """Generates a full, natural, humorous response using the FLAN-T5 model"""
    try:
        result = pipe(
            prompt,
            max_length=150,
            num_return_sequences=1,
            truncation=True,
            do_sample=True,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.2
        )
        raw_output = result[0]["generated_text"].strip()
        return raw_output
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Hmm... I didn't get that. Let's try something else!"

if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=8080)
