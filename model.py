import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Extracted faces on black background. BMP images 128x128
data_folder = './resized_images'

# Load CSV file with height and filenames
csv_path = data_folder + '/final.csv'
data = pd.read_csv(csv_path)

# Define image dimensions
img_size = (128, 128)

# Load and preprocess images
def load_and_preprocess_images(filepaths, labels):
    images = []
    for filepath, label in zip(filepaths, labels):
        filepath = data_folder + '/' + filepath[:-4] + '.bmp'
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return np.array(images), np.array(labels)

# Split the data into training and testing sets
X_train_files, X_test_files, y_train, y_test = train_test_split(
    data['thumbnail_name'].values, data['height_cm'].values, test_size=0.2, random_state=42)

# Load and preprocess training and testing images
X_train, y_train = load_and_preprocess_images(X_train_files, y_train)
X_test, y_test = load_and_preprocess_images(X_test_files, y_test)

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build the model
model = Sequential()
model.add(Input(shape=(img_size[0], img_size[1], 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Print the success rate
success_rate = 100 - mae
print(f'Success rate: {success_rate}%')

# Save the model
model.save('./face_to_height_model_upd.h5')