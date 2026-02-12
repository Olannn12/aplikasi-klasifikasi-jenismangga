"""
FastAPI Backend for Mango Classification
Handles image upload, prediction, and returns results
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import uvicorn
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Mango Classification API",
    description="Deep Learning API for Indonesian Mango Classification",
    version="1.0.0"
)

# CORS Configuration (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    # Path relative to backend folder (when run from backend/)
    MODEL_PATH = "../models/best_model.pth"
    IMG_SIZE = 224
    CLASSES = [
        'mangga_apel',
        'mangga_gedong_gincu',
        'mangga_golek',
        'mangga_harum_manis',
        'mangga_indramayu',
        'mangga_madu',
        'mangga_manalagi'
    ]
    # User-friendly names (optional)
    CLASS_NAMES = {
        'mangga_apel': 'Mangga Apel',
        'mangga_gedong_gincu': 'Mangga Gedong Gincu',
        'mangga_golek': 'Mangga Golek',
        'mangga_harum_manis': 'Mangga Harum Manis',
        'mangga_indramayu': 'Mangga Indramayu',
        'mangga_madu': 'Mangga Madu',
        'mangga_manalagi': 'Mangga Manalagi'
    }

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ##torch disiniii
print(f"Using device: {device}")

# Load model
def load_model():
    """Load the trained MobileNetV3 model"""
    model = models.mobilenet_v3_large(pretrained=False) ## Torchvision disiniii 
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(Config.CLASSES))
    
    # Load trained weights
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully from {Config.MODEL_PATH}")
    print(f"✓ Model accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Global model instance
try:
    model = load_model()
except Exception as e:
    print(f"⚠ Warning: Could not load model: {e}")
    print("API will start but predictions will fail until model is loaded")
    model = None

# Serve static files (frontend) - path relative to backend folder
app.mount("/static", StaticFiles(directory="../static"), name="static")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_path = Path("../static/index.html") ##pathli disiniii
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Mango Classification API</h1><p>Frontend not found. Please add index.html to static folder.</p>")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "classes": len(Config.CLASSES)
    }

@app.get("/api/classes")
async def get_classes():
    """Get list of available mango classes"""
    return {
        "classes": Config.CLASSES,
        "class_names": Config.CLASS_NAMES,
        "total": len(Config.CLASSES)
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict mango variety from uploaded image
    
    Args:
        file: Image file (JPG, PNG, JPEG)
    
    Returns:
        JSON with prediction results
    """
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB') ## import pillow awal
        
        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        top3_predictions = []
        for i in range(3):
            class_id = top3_idx[0][i].item()
            class_name = Config.CLASSES[class_id]
            prob = top3_prob[0][i].item() * 100
            
            top3_predictions.append({
                "class": class_name,
                "class_name": Config.CLASS_NAMES[class_name],
                "confidence": round(prob, 2)
            })
        
        # Main prediction
        predicted_class = Config.CLASSES[predicted.item()]
        confidence_score = confidence.item() * 100
        
        return {
            "success": True,
            "prediction": {
                "class": predicted_class,
                "class_name": Config.CLASS_NAMES[predicted_class],
                "confidence": round(confidence_score, 2)
            },
            "top3": top3_predictions,
            "all_probabilities": {
                Config.CLASS_NAMES[cls]: round(probabilities[0][i].item() * 100, 2)
                for i, cls in enumerate(Config.CLASSES)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict multiple images at once
    
    Args:
        files: List of image files
    
    Returns:
        JSON with batch prediction results
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            # Read and process image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB') ## disini juga pillow 
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = probabilities.max(1)
            
            predicted_class = Config.CLASSES[predicted.item()]
            confidence_score = confidence.item() * 100
            
            results.append({
                "filename": file.filename,
                "success": True,
                "prediction": {
                    "class": predicted_class,
                    "class_name": Config.CLASS_NAMES[predicted_class],
                    "confidence": round(confidence_score, 2)
                }
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total": len(files),
        "results": results
    }

# Run server
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🥭 MANGO CLASSIFICATION API")
    print("="*60)
    print(f"Model: MobileNetV3-Large")
    print(f"Classes: {len(Config.CLASSES)}")
    print(f"Device: {device}")
    print("="*60)
    print("\n🚀 Starting server...")
    print("📱 Open browser: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) ##uvicornn disinii 