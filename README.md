# LeafScan — Plant Disease Detector

**Live Demo:** [https://crop-disease-prediction-1-y2vp.onrender.com](https://crop-disease-prediction-1-y2vp.onrender.com)

LeafScan is an AI-powered web application for detecting plant diseases from leaf images. The application utilizes a fine-tuned **MobileNetV2** model trained on the **PlantVillage dataset**, capable of classifying 38 different plant disease classes (including healthy plants). It also provides AI explainability via **Grad-CAM heatmaps**, highlighting the visual regions the model focused on to make its prediction.

## Project Structure

The project is structured into three main components:

- **`backend/`**: A Python Flask application serving the REST API. It handles image preprocessing, PyTorch model inference, Grad-CAM overlay generation, and prediction logging.
- **`frontend/`**: A clean, responsive web interface built with vanilla HTML, CSS, and JavaScript. It provides a drag-and-drop file upload zone, displays the prediction confidence scores, and visualizes the Grad-CAM heatmap.
- **`model/`**: Contains the pre-trained PyTorch model weights (`best_plant_disease_model.pth`).

## Features

- **Plant Disease Classification**: Detects 38 different plant disease classes across various species (Apple, Corn, Grape, Tomato, Potato, etc.).
- **Explainable AI (Grad-CAM)**: Visualizes the model's focus area using a heatmap overlay to explain why a specific prediction was made.
- **Top 5 Predictions**: Returns the top 5 most likely classes with their confidence probabilities.
- **Prediction Logging**: Stores historical prediction logs locally in JSON format.
- **Drag-and-Drop Interface**: Easy-to-use frontend for uploading leaf images.

## Technology Stack

- **Backend**: Python, Flask, PyTorch, Torchvision, Pillow, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Model**: MobileNetV2 (Pretrained and Fine-tuned)

## Running the Project Locally

### 1. Setup the Backend

1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask development server:
   ```bash
   python app.py
   ```
   The backend will start on `http://127.0.0.1:5000`.

### 2. Run the Frontend

The frontend consists of static files. You can open `frontend/index.html` directly in your web browser, or use a simple HTTP server to serve the files:

```bash
cd frontend
# Using Python's built-in HTTP server:
python -m http.server 8000
```
Then, open `http://localhost:8000` in your web browser.

## API Endpoints

### `GET /`
Returns the status of the API and model information.

### `POST /predict`
Accepts an image file (`multipart/form-data` with key `file` or `image`) or a base64 encoded image string via JSON (`{"image": "..."}`).

**Returns:**
- `prediction`: The predicted class name.
- `confidence`: The confidence score of the top prediction.
- `is_healthy`: Boolean indicating if the plant is healthy.
- `top5`: List of the top 5 predictions with labels and confidence scores.
- `gradcam_overlay`: A Base64 encoded PNG string of the Grad-CAM heatmap overlay.
