# Celebrity Height Prediction System

This system utilizes the DeepFace library for face detection and analysis, and a custom Keras convolutional neural network (CNN) model to predict the heights of a person based on their facial features. The system consists of several scripts for data collection, preprocessing, model training, and height prediction.

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
- The model is designed to predict celebrity heights from resized face images.
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

