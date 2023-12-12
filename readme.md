# Person Height Prediction System

This system designed for predicting the age, gender, race, emotion and height of individuals using a combination of web scraping, face detection, and deep learning techniques. The system starts by collecting celebrity data from 'https://www.celebheights.com/' through the celebheights.py script. The gathered data undergoes a cleaning process in cleaner.py, followed by resizing images in resizer.py based on detected facial areas. A convolutional neural network (CNN) model is then built, trained, and evaluated in model.py to predict person heights.

The run.py script integrates face recognition with the trained model, providing a comprehensive solution to predict heights and analyze various facial attributes of individuals. This repository aims to showcase the entire process from data collection to prediction, providing insights into the development of a person height prediction system.

## Scripts

### 1. `celebheights.py`

- Script for scraping celebrity data, including names, links, heights, and photo URLs, from the "https://www.celebheights.com/" website.
- Downloads celebrity photos and saves them along with metadata in CSV format.

### 2. `cleaner.py`

- Uses DeepFace library for face detection and verification.
- Removes unwanted faces based on predefined images.
- Outputs face coordinates and confidence in CSV format.

### 3. `resizer.py`

- Resizes face images based on detected facial areas.
- Saves resized images in a new directory.

### 4. `model.py`

- Creates, trains, and evaluates a CNN model using TensorFlow/Keras.
- The model is designed to predict person height from resized face images.
- Saves the trained model.

### 5. `run.py`

- Performs face recognition on input images using DeepFace library.
- Utilizes the trained model to predict heights.
- Outputs visualizations and a CSV file with detailed information about detected faces.

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/DrZuse/ML-face2height.git
    cd ML-face2height
    ```

2. Install the required dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run `celebheights.py` to collect celebrity data.
2. Run `cleaner.py` to clean unwanted faces.
3. Run `resizer.py` to resize face images.
4. Run `model.py` to train the height prediction model.
5. Place input images in the `input` directory.
6. Run `run.py` to perform face recognition and predict heights.

## Dependencies

- Python 3.x
- Libraries: requests, bs4, deepface, OpenCV, NumPy, Pandas, Pillow, TensorFlow

