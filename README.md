# MedVision AI — Professional Chest X-Ray Website

This project wraps the trained pneumonia CNN and Grad-CAM pipeline in a professional HTML/CSS/JavaScript + Flask website.

## Setup

1. Put your trained model here:
   `models/pneumonia_cnn_model.keras`

2. Install dependencies:
   `pip install -r requirements.txt`

3. Start the website:
   `python app.py`

4. Open:
   `http://127.0.0.1:5000`

## Project structure

- `app.py` — Flask backend, model inference and Grad-CAM
- `templates/index.html` — website
- `static/css/style.css` — professional responsive design
- `static/js/app.js` — upload and prediction logic
- `models/` — place your trained Keras model here

## Important

This is an educational/research prototype. It must not be presented as a clinically validated diagnostic system.
