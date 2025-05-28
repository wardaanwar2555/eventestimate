from flask import Flask, request, render_template, jsonify
import os
import requests
import base64
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
GENKIT_API_KEY = os.getenv("GENKIT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Simplified implementation of the mood board generation functionality
async def generate_mood_board(theme_description):
    """
    Generate a mood board image based on the provided theme description
    using Google's Gemini API.
    
    Args:
        theme_description (str): Description of the desired theme and mood
        
    Returns:
        dict: Contains the mood board image URL as base64 data
    """
    # API endpoint for Gemini
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {"text": theme_description}
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_LOW_AND_ABOVE"
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        # Extract the image data from the response
        for part in result["candidates"][0]["content"]["parts"]:
            if "inlineData" in part:
                image_data = part["inlineData"]["data"]
                return {"moodBoardImageUrl": f"data:image/png;base64,{image_data}"}
    
    # Return error if generation failed
    return {"error": "Failed to generate mood board image"}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    theme_description = data.get('themeDescription', '')
    
    if not theme_description:
        return jsonify({"error": "Theme description is required"}), 400
    
    # Generate the mood board
    result = generate_mood_board(theme_description)
    
    return jsonify(result)

# Create templates directory and add basic template
if not os.path.exists('templates'):
    os.makedirs('templates')

if __name__ == '__main__':
    app.run(debug=True)