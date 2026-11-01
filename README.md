# AI-Smart-Street-Light
Webcam-based day/night detection using Teachable Machine and TensorFlow Lite that automatically controls street lights through an Arduino UNO.
# 💡 AI Smart Street Light

An AI-powered street light system that detects whether it is day or night using a camera and automatically turns the lights on or off.

## Overview

Traditional street lights run on fixed timers or simple light sensors, which can waste energy or fail in changing conditions. This project uses computer vision instead. A camera watches the scene, an AI model decides if it is day or night, and an Arduino switches the lights accordingly.

## How It Works

1. **Capture:** A webcam continuously records the surroundings.
2. **Classify:** A model trained with Teachable Machine and run through TensorFlow Lite labels each frame as DAY or NIGHT.
3. **Verify:** The system acts only when the prediction is confident and stays consistent over several frames, which prevents flickering from sudden shadows or passing lights.
4. **Control:** The result is sent to an Arduino UNO, which turns the lights OFF during the day and ON at night.

## Key Features

- Real-time day and night detection
- Confidence threshold and stability check for reliable decisions
- Automatic Arduino detection
- Live on-screen display showing status, confidence, and connection state
- Lights switch off safely when the program closes

## Benefits

- Saves energy by lighting only when needed
- Adapts to real conditions instead of fixed schedules
- Low cost and easy to build with common hardware

## Built With

Python · OpenCV · TensorFlow Lite · Teachable Machine · Arduino UNO
