import os
import time
import random
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
import io


app = Flask(__name__)
CORS(app)

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

def generate_question_suggestion():
    suggestions = [
        "What are some simple exercises I can do at home?",
        "How can I improve my sleep quality?",
        "What are the signs of dehydration I should watch out for?",
        "Can you explain the importance of regular health check-ups?",
        "What are some heart-healthy foods I should include in my diet?",
        "How can I manage stress effectively?",
        "What are the common side effects of my medications?",
        "How can I maintain good hygiene to prevent infections?",
        "What are some memory exercises I can practice daily?",
        "How can I make my home safer to prevent falls?",
        "What should I include in a balanced meal for my age?",
        "How often should I have my vision and hearing checked?",
        "What are some social activities suitable for seniors in my area?",
        "How can I keep my brain active and healthy?",
        "What are the benefits of staying socially active in old age?"
    ]
    return random.choice(suggestions)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        input_text = msg
        print(f"Received message: {input_text}")

        time.sleep(1)

        response = model.generate_content(input_text)
        generated_text = response.text

        print(f"Response: {generated_text}")

        return jsonify({"response": generated_text})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again."}), 500

@app.route("/get_suggestion", methods=["GET"])
def get_suggestion():
    try:
        suggestion = generate_question_suggestion()
        return jsonify({"suggestion": suggestion})
    except Exception as e:
        print(f"Error generating suggestion: {e}")
        return jsonify({"error": "Unable to generate a suggestion at this time."}), 500

@app.route("/voice_input", methods=["POST"])
def voice_input():
    try:
        data = request.json
        voice_input_text = data["voice_input"]
        print(f"Voice input received: {voice_input_text}")

        time.sleep(1)

        response = model.generate_content(voice_input_text)
        generated_text = response.text

        print(f"Response: {generated_text}")

        return jsonify({"response": generated_text})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing your voice input. Please try again."}), 500

@app.route("/process_input", methods=["POST"])
def process_input():
    try:
        msg = request.form.get("msg", "")
        image = request.files.get("image")

        if image:
            img = Image.open(io.BytesIO(image.read()))
            response = vision_model.generate_content([msg, img])
        else:
            response = model.generate_content(msg)

        generated_text = response.text
        print(f"Response: {generated_text}")

        return jsonify({"response": generated_text})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again."}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
