import os, cv2, csv
import numpy as np
from PIL import Image
from deepface import DeepFace
import keras_core as keras


def output_data(output, file, output_obj, image):
    # creating a csv file with header
    with open(f'{output}/{file}.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'file', 'face_id', 'face_rating', 'age', 
            'gender', 'race', 'emotion', 'height'])

    for i, f in enumerate(output_obj):
        face_id = f'ID_{i}'
        x, y, w, h = f['facial_area'].values()

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, face_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(f'{output}/{file}', image)

        with open(f'{output}/{file}.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                file, face_id, f['confidence'], f['age'], 
                f['dominant_gender'], f['dominant_race'], 
                f['dominant_emotion'], f['height']])


def face_recognition(input, output):
    for file in os.listdir(input):
        input_img_path = os.path.join(input, file)

        try:
            image = cv2.imread(input_img_path)
            detections = DeepFace.extract_faces(
                image,
                detector_backend = 'retinaface')

        except Exception as e:
            print(str(e))

        else:
            output_obj = []
            for d in detections:
                x, y, w, h = d['facial_area'].values()
                detected_face = image[y:y+h, x:x+w]
                
                face_analyze = DeepFace.analyze(
                    img_path = detected_face,
                    enforce_detection = False,
                    detector_backend = 'skip',
                    actions = ['age', 'gender',
                            'race', 'emotion'])

                face_analyze[0]['confidence'] = d['confidence']
                face_analyze[0]['facial_area'] = d['facial_area']

                # load the model to make predictions
                loaded_model = keras.saving.load_model('face_to_height_model.h5')

                # Convert the region array back to an image
                # [:, :, ::-1] - BGR2RGB
                region_image = Image.fromarray(detected_face[:, :, ::-1])

                old_size = region_image.size
                # calculate the new size maintaining the aspect ratio
                ratio = min(128.0 / old_size[0], 128.0 / old_size[1])
                new_size = tuple([int(x*ratio) for x in old_size])

                # resize the image
                resized_img = region_image.resize(new_size, Image.ANTIALIAS)

                # create a new image with a black background
                new_img = Image.new('RGB', (128, 128), 'black')
                new_img.paste(resized_img, ((128-new_size[0])//2, (128-new_size[1])//2))

                output_array = np.array(new_img) / 255.0  # Normalize pixel values
                output_array = np.expand_dims(output_array, axis=0)  # Add batch dimension

                # Use the loaded model to make predictions
                predicted_height = loaded_model.predict(output_array)[0][0]
                face_analyze[0]['height'] = predicted_height

                output_obj.extend(face_analyze)

            output_data(output, file, output_obj, image)


if __name__ == "__main__":

    input = './input'
    output = './output'

    if not os.path.exists(output):
        os.makedirs(output)

    face_recognition(input, output)