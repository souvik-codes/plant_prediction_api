import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore

# Define the FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],   # Allow all headers
)

# Load the model
model = load_model('my_model.h5')

# Define the plant names
names = [
    'Aloe Vera', 'Amla', 'Amruta Balli', 'Arali', 'Ashoka', 'Ashwagandha',
    'Avocado', 'Bamboo', 'Basale', 'Betel', 'Betel Nut', 'Brahmi', 'Castor',
    'Curry Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Guava', 'Geranium',
    'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon',
    'Lemon Grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa',
    'Nooni', 'Papaya', 'Pepper', 'Pomegranate', 'Raktachandini',
    'Rose', 'Sapota', 'Tulsi', 'Wood Sorel'
]

# Preprocess image function using OpenCV
def preprocess_image(image):
    # Decode the image
    image_array = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Resize to 224x224
    img_resized = cv2.resize(img, (224, 224))
    
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Expand dimensions to match model input shape
    return np.expand_dims(img_rgb, axis=0)

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = await file.read()
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)
        
        # Convert to one-hot encoded format
        one_hot_encoded = np.zeros(len(names))
        one_hot_encoded[predicted_class] = 1
        
        description, uses = get_plant_info(names[predicted_class[0]])

        return JSONResponse(content={
            "predictions": one_hot_encoded.tolist(),
            "predicted_label": names[predicted_class[0]],
            "description": description,
            "uses": uses
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


def get_plant_info(plant_name):
    # Load the JSON data from the file
    with open('plants.json', 'r') as file:
        data = json.load(file)

    # Search for the plant in the data
    for plant in data['plants']:
        if plant['name'].lower() == plant_name.lower():
            return plant['description'], plant['uses']

    return None, None


# Run the application
#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)