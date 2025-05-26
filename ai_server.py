from transformers import pipeline, set_seed
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the LOCAL model (no internet needed after download)
MODEL_PATH = "./local_flan_t5_small"
pipe = pipeline("text-generation", model=MODEL_PATH, framework="pt")

app = Flask(__name__)
CORS(app)

# Intent-to-route mapping
INTENT_MAP = {
    "home": {
        "prompt": "Explain going to the home page in a funny way using Indian English.",
        "route": "/"
    },
    "about pitch": {
        "prompt": "Describe navigating to About Pitch page like a confused college lecturer.",
        "route": "/about/pitch"
    },
    "about team": {
        "prompt": "Explain About Team page like a local event compere announcing the stars.",
        "route": "/about/team"
    },
    "contact us": {
        "prompt": "Open Contact Us page like an overenthusiastic government office peon.",
        "route": "/contact"
    },
    "courses": {
        "prompt": "Show available courses like a desperate coaching center guy trying to impress parents.",
        "route": "/courses"
    },
    "features": {
        "prompt": "Take them to features page like a local tech support guy explaining rocket science.",
        "route": "/features"
    },
}

def generate_response(prompt):
    """Generates a single-line humorous response using the local GPT-2 model"""
    try:
        result = pipe(
            prompt,
            max_length=50,
            num_return_sequences=1,
            truncation=True,         # <-- Added to suppress warning
            pad_token_id=50256       # <-- Set explicitly for gpt2 models
        )
        raw_output = result[0]["generated_text"]
        return raw_output.strip().split("\n")[0]
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I had trouble understanding that. Try again."

@app.route("/bot", methods=["GET"])
def bot_query():
    user_query = request.args.get("query", "").strip().lower()
    print(f"User said: {user_query}")

    matched_intent = None
    for key in INTENT_MAP:
        if key in user_query:
            matched_intent = INTENT_MAP[key]
            break

    if matched_intent:
        prompt = f"In a funny way: {matched_intent['prompt']}"
        clean_response = generate_response(prompt)

        return jsonify({
            "response": clean_response,
            "route": matched_intent["route"]
        })

    else:
        prompt = f"hear me out: '{user_query}'"
        clean_response = generate_response(prompt) + " Try something else."

        return jsonify({
            "response": clean_response,
            "route": None
        })


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)
