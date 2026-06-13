# Cataract Severity Classification Using ResNet50

A web-based deep learning system for classifying cataract severity levels from eye images using Transfer Learning with ResNet50.

## Overview

This project aims to automatically classify cataract severity into three categories:

- Normal
- Immature Cataract
- Mature Cataract

The model is developed using TensorFlow/Keras with a ResNet50 architecture pre-trained on ImageNet and fine-tuned for cataract image classification. A Streamlit web application is provided to enable real-time image-based predictions.

## Dataset

Dataset: Eye Cataract (Mature, Immature, Normal)

Classes:

- Cataract Mature
- Cataract Immature
- Normal

## Methodology

### Data Preprocessing

- Image resizing to 224 × 224 pixels
- ResNet50 preprocessing
- Data augmentation:
  - Rotation
  - Zoom
  - Brightness adjustment
  - Width and height shifting
  - Horizontal flipping
  - Shearing

### Model Architecture

- Base Model: ResNet50 (ImageNet pretrained)
- Global Average Pooling
- Dense (256 units, ReLU)
- Batch Normalization
- Dropout (0.5)
- Output Layer (Softmax)

### Training Strategy

- Transfer Learning
- Frozen base layers
- Adam optimizer
- Early Stopping
- Reduce Learning Rate on Plateau
- Model Checkpoint

## Technologies

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

## Project Structure

```text
.
├── app.py
├── best_model.keras
├── requirements.txt
└── README.md
```

## Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal and upload an eye image to obtain a prediction.

## Output

The application provides:

- Predicted class
- Confidence score
- Probability distribution for all classes

## Disclaimer

This application is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

## Author

Dewi Safira
