# Wall Detection AI

Wall Detection AI is a computer vision based web application capable of detecting indoor wall boundaries using a custom trained YOLOv8 segmentation model.

The system allows users to:

- Upload indoor room images
- Detect wall boundaries automatically
- Generate contour-based wall segmentation
- Visualise predictions directly in a modern web interface

The project combines:

- YOLOv8 Segmentation
- ADE20K Dataset
- FastAPI Backend
- Modern Frontend UI
- OpenCV Image Processing

---

# Project Objective

The goal of this project is to create an AI-powered wall segmentation system capable of accurately identifying wall regions in indoor environments.

This serves as the foundation for future features such as:

- Wallpaper visualisation
- Interior redesign
- Wall texture projection
- Smart room editing
- AR-based home customisation

Currently, the model focuses ONLY on:

- Wall detection
- Wall boundary segmentation

The model does NOT attempt to detect:

- Ceilings
- Floors
- Furniture
- Objects

This was an intentional design decision in order to maintain focused segmentation behaviour.

---

# Technologies Used

Frontend:
- HTML
- CSS
- JavaScript

Backend:
- FastAPI
- Uvicorn

AI / Computer Vision:
- YOLOv8 Segmentation
- OpenCV
- NumPy
- Ultralytics

Dataset:
- ADE20K Indoor Scene Dataset

Training Environment:
- Google Colab GPU

Development Environment:
- VS Code
- macOS (Apple Silicon M4)

---

# Development Process

The system was developed step-by-step using the following pipeline.

---

## 1. Dataset Preparation

The ADE20K dataset was downloaded and filtered to contain ONLY indoor scenes.

Indoor environments such as:
- bedrooms
- bathrooms
- living rooms
- kitchens
- offices

were selected.

The ADE20K wall class ID was extracted and used to create binary wall masks.

These masks represented:
- white = wall
- black = non-wall

---

## 2. Wall Mask Generation

Binary wall masks were generated using OpenCV.

The processing pipeline:
- read segmentation masks
- isolated wall pixels
- created binary wall-only masks
- stored processed masks

This created a cleaner segmentation target for training.

---

## 3. YOLO Segmentation Dataset Creation

The binary masks were converted into YOLO segmentation polygon format.

The dataset was then split into:
- training
- validation

directories.

YOLO segmentation labels were generated using polygon contour extraction.

---

## 4. Model Training

The model was trained using:

- YOLOv8 Segmentation
- Google Colab GPU

The model currently has been trained for ONLY:

- 20 epochs

This is important.

The current results are already functional and visually impressive, however:

Training for:
- more epochs
- longer durations
- larger datasets
- cleaner masks

would significantly improve:
- accuracy
- contour precision
- segmentation quality
- generalisation

Current model quality should therefore be considered:
- an early-stage trained model
- not a fully optimised production model

---

## 5. Contour Extraction Pipeline

After inference, the raw segmentation masks were post-processed using OpenCV.

This pipeline:
- cleaned masks
- removed noise
- extracted contours
- approximated wall polygons
- rendered wall boundaries

The final system draws ONLY wall outlines rather than filling entire wall regions.

This produced significantly cleaner and more visually appealing results.

---

## 6. Backend Development

A FastAPI backend was developed to:
- receive uploaded images
- run YOLO inference
- process segmentation masks
- return prediction images

The backend serves as the AI inference engine of the application.

---

## 7. Frontend Development

A modern frontend interface was created featuring:
- glassmorphism UI
- neon gradients
- animated glowing background
- drag-and-drop upload area
- real-time prediction preview

The frontend communicates with the FastAPI backend using JavaScript fetch requests.

---

# Current Features

- Indoor wall boundary detection
- Image upload system
- Real-time AI prediction
- Contour-based segmentation
- Modern responsive UI
- Fast inference pipeline

---

# Future Improvements

Potential future improvements include:

- Improved training with more epochs
- Better segmentation precision
- Wallpaper visualisation
- Texture replacement
- Multi-wall selection
- Real-time webcam support
- React frontend
- Cloud deployment
- Mobile responsiveness
- User accounts
- Image history

---

# Project Structure

wall-detection-ai/

├── backend/
├── frontend/
├── inference/
├── models/
├── scripts/
├── training/
├── uploads/
├── processed/
└── README.md

---

# Running the Project

## 1. Start Backend

From project root:

python -m uvicorn backend.app:app --reload

Backend runs on:

http://127.0.0.1:8000

---

## 2. Start Frontend

Open frontend directory:

cd frontend

Run local server:

python3 -m http.server 5500

Frontend runs on:

http://127.0.0.1:5500

---

# Model Information

Current model:
- YOLOv8 Segmentation

Current training:
- 20 epochs

Important note:
The model is still capable of further improvement with additional training and dataset refinement.

---

# Important Notes

- The project currently supports indoor scenes only.
- Wall detection is the primary and only segmentation target.
- The model is not yet production-optimised.
- GPU acceleration is recommended for training.
- CPU inference is still fast enough for real-time interaction.

---

# Author

Developed as a custom AI wall segmentation system using computer vision and modern web technologies.