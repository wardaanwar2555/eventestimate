from flask import Flask, render_template, request, jsonify, redirect, url_for
import google.generativeai as genai
import pandas as pd
import numpy as np
import joblib
import os
import base64
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Configure Google Generative AI
genai.configure(api_key="AIzaSyCfDwGhorgHVj7UFpoqAiAt2B6yG5HsHqs")  # Replace with your actual API key

# Model configuration for Gemini
MODEL_CONFIG = {
    "model": "gemini-2.0-flash-exp",
    "generation_config": {
        "response_mime_type": "image/png",
        "temperature": 0.7,
        "max_output_tokens": 2048
    }
}

# Load the trained model or train if it doesn't exist
MODEL_PATH = 'models/event_budget_model.pkl'

# Check if model directory exists, otherwise create it
if not os.path.exists('models'):
    os.makedirs('models')

# Function to train and save the model if it doesn't exist
def train_and_save_model():
    print("Training model since it doesn't exist...")
    
    # Load dataset
    df = pd.read_csv("event_budget_dataset.csv")
    
    # Feature Engineering
    df['Total_Food_Cost'] = df['Attendees'] * df['Food_Cost_Per_Person']
    
    # Feature Selection
    features = ['Attendees', 'Venue_Cost', 'Food_Cost_Per_Person',
                'Music_System_Cost', 'Lighting_Cost', 'Total_Food_Cost']
    target = 'Total_Budget'
    X = df[features]
    y = df[target]
    
    # Train model (using full dataset)
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved successfully!")
    return model

# Load or train the model
if not os.path.exists(MODEL_PATH):
    event_model = train_and_save_model()
else:
    print("Loading existing model...")
    event_model = joblib.load(MODEL_PATH)

# Mood Board Generation Function
async def generate_mood_board(theme_description):
    try:
        # Configure generative model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Generate response with image
        response = await model.generate_content_async(
            theme_description,
            generation_config={
                "response_mime_type": "image/png",
            },
            stream=False
        )
        
        # Extract image data
        if response.parts and hasattr(response.parts[0], 'image'):
            image_data = response.parts[0].image.data
            # Convert binary data to base64 for embedding in HTML
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
        else:
            raise Exception("No image generated in the response")
    except Exception as e:
        print(f"Error generating mood board: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/eventtypes')
def event_types():
    return render_template('eventtypes.html')

@app.route('/Estimation')
def estimate():
    return render_template('Estimation.html')

@app.route('/about')
def about():
    return render_template('About.html')

MODEL_CONFIG = {
    "model": "models/gemini-1.5-flash",  # or your specific model
    "generation_config": {
        "temperature": 0.7,
        "max_output_tokens": 256,
    }
}

@app.route('/aigenerate', methods=['GET', 'POST'])
def aigenerate():
    generated_text = None
    error = None

    if request.method == 'POST':
        prompt = request.form.get('prompt')

        if prompt:
            try:
                # Initialize the Gemini model
                gemini_flash = genai.GenerativeModel(
                    model_name=MODEL_CONFIG["model"],
                    generation_config=genai.types.GenerationConfig(
                        **MODEL_CONFIG["generation_config"]
                    )
                )

                # Generate content from the prompt
                response = gemini_flash.generate_content(prompt)

                # Extract text safely from the response
                if hasattr(response, 'text') and response.text:
                    generated_text = response.text.strip()
                else:
                    error = "No text was returned by the model."

            except Exception as e:
                error = f"Error generating text: {str(e)}"

    return render_template('aigenerate.html', generated_text=generated_text, error=error)
@app.route('/getdemo', methods=['GET', 'POST'])
def getdemo():
    estimation_result = None
    
    if request.method == 'POST':
        try:
            # Get form data
            attendees = int(request.form.get('attendees', 0))
            venue_cost = float(request.form.get('venue_cost', 0))
            food_cost_per_person = float(request.form.get('food_cost_per_person', 0))
            music_system_cost = float(request.form.get('music_system_cost', 0))
            lighting_cost = float(request.form.get('lighting_cost', 0))
            
            # Calculate total food cost
            total_food_cost = attendees * food_cost_per_person
            
            # Prepare input data for prediction
            input_data = [[
                attendees, 
                venue_cost, 
                food_cost_per_person, 
                music_system_cost, 
                lighting_cost, 
                total_food_cost
            ]]
            
            # Make prediction
            estimated_budget = event_model.predict(input_data)[0]
            
            # Prepare cost breakdown
            cost_breakdown = {
                "Venue Cost": venue_cost,
                "Food Cost": total_food_cost,
                "Music System Cost": music_system_cost,
                "Lighting Cost": lighting_cost
            }
            
            # Create result dictionary
            estimation_result = {
                "estimated_total_budget": round(estimated_budget, 2),
                "cost_breakdown": cost_breakdown
            }
            
        except Exception as e:
            estimation_result = {"error": str(e)}
    
    return render_template('getdemo.html', estimation_result=estimation_result)

@app.route('/moodboard', methods=['GET', 'POST'])
def mood_board():
    image_url = None
    theme_description = None
    
    if request.method == 'POST':
        theme_description = request.form.get('themeDescription', '')
        if theme_description:
            import asyncio
            # Run the async function in a synchronous context
            image_url = asyncio.run(generate_mood_board(theme_description))
    
    return render_template('moodboard.html', image_url=image_url, theme_description=theme_description)

# Add an API endpoint for programmatic access to budget estimation
@app.route('/api/estimate', methods=['POST'])
def api_estimate_budget():
    try:
        # Get JSON data
        data = request.get_json()
        
        attendees = int(data['attendees'])
        venue_cost = float(data['venue_cost'])
        food_cost_per_person = float(data['food_cost_per_person'])
        music_system_cost = float(data['music_system_cost'])
        lighting_cost = float(data['lighting_cost'])
        
        # Calculate total food cost
        total_food_cost = attendees * food_cost_per_person
        
        # Prepare input data for prediction
        input_data = [[
            attendees, 
            venue_cost, 
            food_cost_per_person, 
            music_system_cost, 
            lighting_cost, 
            total_food_cost
        ]]
        
        # Make prediction
        estimated_budget = event_model.predict(input_data)[0]
        
        # Prepare cost breakdown
        cost_breakdown = {
            "Venue Cost": venue_cost,
            "Food Cost": total_food_cost,
            "Music System Cost": music_system_cost,
            "Lighting Cost": lighting_cost
        }
        
        # Return the result
        return jsonify({
            "success": True,
            "estimated_total_budget": round(estimated_budget, 2),
            "cost_breakdown": cost_breakdown
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# Add an API endpoint for mood board generation
@app.route('/api/moodboard', methods=['POST'])
def api_generate_mood_board():
    try:
        # Get JSON data
        data = request.get_json()
        theme_description = data.get('themeDescription', '')
        
        if not theme_description:
            return jsonify({
                "success": False,
                "error": "Theme description is required"
            })
        
        import asyncio
        # Generate mood board
        image_url = asyncio.run(generate_mood_board(theme_description))
        
        if image_url:
            return jsonify({
                "success": True,
                "moodBoardImageUrl": image_url
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to generate mood board"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# Add an API endpoint for image generation
@app.route('/api/generate-image', methods=['POST'])
def api_generate_image():
    try:
        # Get JSON data
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({
                "success": False,
                "error": "Prompt is required"
            })
        
        # Initialize the image generation model with Gemini Flash 2.0
        gemini_flash = genai.GenerativeModel(
            MODEL_CONFIG["model"],
            generation_config=genai.types.GenerationConfig(
                **MODEL_CONFIG["generation_config"]
            )
        )
        
        # Generate image using Gemini Flash 2.0
        response = gemini_flash.generate_content(prompt)
        
        # Process the response
        image_data = None
        if hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                if hasattr(part, 'image') and part.image:
                    # Convert image data to base64 format
                    image_bytes = part.image.data
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    image_data = f"data:image/png;base64,{image_base64}"
                    break
        
        if image_data:
            return jsonify({
                "success": True,
                "imageData": image_data
            })
        else:
            return jsonify({
                "success": False,
                "error": "No image was generated in the response"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)