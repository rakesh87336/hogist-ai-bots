from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# LM Studio Server URL
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "gemma-2-2b-it-Q5_K_M.gguf"  # Must match the loaded model name in LM Studio

# Intent-to-route mapping with funny prompts
INTENT_MAP = {
    "home": {
        "prompt": "Welcome to the Home page like a friendly local tea stall owner who just discovered emojis ☕.",
        "route": "/"
    },
    "about": {
        "prompt": "Explain About page like a confused but enthusiastic tour guide lost in his own city.",
        "route": "/about"
    },
    "contact": {
        "prompt": "Open Contact Us page like an overenthusiastic government office peon who loves exclamation marks!!!",
        "route": "/contact"
    },
    "features": {
        "prompt": "Describe Features page like a local tech support guy explaining rocket science using only food analogies.",
        "route": "/features"
    },
    "menu": {
        "prompt": "Take them to Menu page like a loud roadside food vendor shouting over traffic.",
        "route": "/menu"
    },
    "services": {
        "prompt": "Go to Services page like a motivational speaker selling life-changing ideas using only memes.",
        "route": "/services"
    }
}

# Knowledge-based Q&A about Hogist
KNOWLEDGE_RESPONSES = {
    "who is hogist": """
        Founded in 2018, HOGIST is India’s leading bulk food aggregator platform.
        We deliver unified solutions for food and beverage needs across the country,
        leveraging cutting-edge technology to ensure a seamless experience in bulk food ordering and delivery. 🍛🚀
    """,
    "what does hogist do": """
        HOGIST offers a wide range of services including Industrial Catering, Corporate Events,
        Food Court Setup, Homemade Food Subscriptions, and more. We focus on hygiene, quality,
        and consistent service for both corporates and individuals. Basically, we feed your office without drama.
    """,
    "vision of hogist": """
        Our vision is to become the premier marketplace for bulk food solutions—providing hygienic,
        high-quality meals with exceptional service. We aim to make a positive impact through food,
        empower our team, and build lasting culinary memories. Also, we want every bite to feel like home.
    """,
    "mission of hogist": """
        We are committed to:
        - Providing delicious, hygienic food at affordable prices.
        - Delivering outstanding, consistent, and flexible service.
        - Creating memorable experiences by prioritizing customer delight.
        In short: Tasty food, no fuss, always fresh!
    """,
    "core values": """
        Our Core Values include:
        - EMPLOYEE FIRST: Safe, productive, and rewarding workplace.
        - ETHICS: Integrity and fairness in everything we do.
        - EXTRA MILE: Exceed expectations in hygiene, taste, and service.
        - ENTERTAINMENT: Create joyful food experiences, not just meals.
        We're not just feeding people — we're entertaining them too.
    """,
    "services offered": """
        HOGIST offers:
        - Corporate & Industrial Catering
        - Event Catering (Weddings, Conferences, Trade Shows)
        - Subscription Meal Plans
        - Homemade Food Delivery
        - Food Court & Canteen Setup
        - Mobile Kitchens (Food Trucks/Bikes)
        From office lunches to wedding feasts, we've got your plate covered.
    """,
    "products": """
        Key products of HOGIST include:
        - Nutrition-Focused Meal Plans
        - Diabetic-Friendly Meals
        - Cost Plus Pricing Model (Transparent margins)
        - AI-Powered Aggregator Platform
        We bring together food lovers, chefs, and data geeks in one tasty app.
    """,
    "pricing model": """
        HOGIST uses a transparent pricing model where clients pay only for the cost of food + fixed margin.
        We offer cost optimization without compromising quality, along with customization and data-driven insights.
        No hidden fees, just good food and clear bills.
    """,
    "why hogist": """
        What sets us apart:
        - Biannual food & water testing
        - ISO-certified kitchens and practices
        - App-based order tracking and QR feedback system
        - Dynamic menu updates
        We don’t just cook — we innovate, test, and perfect every meal.
    """,
    "clients": """
        Trusted by industry giants such as Johnson Electric, Siemens Gamesa, Fuji Electric, Equitas Bank,
        Holiday Inn, and many more. We're proud to serve corporates and institutions nationwide.
        If they're big and hungry, we've probably fed them already.
    """,
    "contact": """
        For contact details, please visit the /contact route or reach out to:
        Email: support@hogist.com
        Corporate Office: Chennai, Tamil Nadu
        Or just follow the smell of freshly cooked biryani — it'll lead you right here!
    """
}

# Store recent logs
request_logs = []

@app.before_request
def log_request_info():
    log_entry = f"{request.remote_addr} - [{request.method}] {request.url}"
    print(log_entry)
    request_logs.append(log_entry)
    if len(request_logs) > 50:  # Limit log size
        request_logs.pop(0)

@app.route("/")
def index():
    logs_html = "<br>".join(request_logs[-10:])  # Show last 10 logs
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Hogist AI Server</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }
            h1 { color: #333; }
            pre { background: #fff; padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1); overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>🎉 The Hogist AI Server is Running!</h1>
        <p><strong>Address:</strong> http://localhost:8080</p>
        <hr>
        <h2>📜 Recent Request Logs:</h2>
        <pre>{{ logs }}</pre>
    </body>
    </html>
    """
    return render_template_string(html, logs=logs_html)

@app.route("/bot", methods=["GET"])
def bot_query():
    user_query = request.args.get("query", "").strip().lower()
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
        return jsonify({
            "response": clean_response,
            "route": matched_intent["route"],
        })

    # Check for knowledge-based question
    for key in KNOWLEDGE_RESPONSES:
        if key in user_query:
            response_text = KNOWLEDGE_RESPONSES[key].strip()
            return jsonify({
                "response": response_text
            })

    # Fallback general query
    clean_response = generate_response(user_query) + " Still not sure what you mean? Ask me anything!"
    return jsonify({
        "response": clean_response
    })


def generate_response(prompt):
    """Generates a full, natural response using the LM Studio hosted Gemma model"""
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.95,
            "top_p": 0.9,
            "max_tokens": 200
        }

        response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['message']['content'].strip()
            return generated_text
        else:
            print(f"Error from LM Studio: {response.status_code}, {response.text}")
            return "I'm having trouble thinking right now. Let’s try again later!"
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Hmm... I didn't get that. Let's try something else!"


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)