from deepface import DeepFace
import cv2, numpy, csv, os

# Directory containing all photos with faces
data_folder = './celebheights'
# Directory containing unwanted faces
delf = './avoid_faces'

# Creating a csv file with header
with open(data_folder+'/detections.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['thumbnail_name', 'x', 'y', 'w', 'h', 'confidence'])

# Setting the detector backend
detector = 'retinaface'

# List to store unwanted faces
avoid_faces_list = []
for filename in os.listdir(delf):
    img_path = os.path.join(delf, filename)
    img = cv2.imread(img_path)
    width = 128
    avoid_faces_list.append(cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1]))))

# Converting list to numpy array
avoid_faces_arr = numpy.vstack(avoid_faces_list)

# Function to extract face coordinates
def face_coordinats(filename):
    img_path = os.path.join(data_folder, filename)
    print(img_path)

    try:
        # Extracting faces from the image
        detections = DeepFace.extract_faces(
            img_path,
            detector_backend = detector)
        print(f'total detections: {len(detections)}')
    except Exception as e:
        print(str(e))
        return

    # If more than one face is detected
    if len(detections) > 1:
        # Verifying the faces
        result = DeepFace.verify(
            img1_path = img_path,
            img2_path = avoid_faces_arr,
            distance_metric = 'euclidean',
            model_name = 'Facenet',
            detector_backend = detector)

        # If faces are verified
        if result['verified']:
            for i, l in enumerate(detections):
                if l['facial_area'] == result['facial_areas']['img1']:
                    del detections[i]

    # Filtering detections based on confidence
    detections = [x for x in detections if x['confidence'] > 0.9985] if len(detections) > 1 else detections

    # If only one face is detected
    if len(detections) == 1:
        x, y, w, h = detections[0]['facial_area'].values()
        with open(data_folder+'/detections.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([filename, x,y,w,h, detections[0]['confidence']])

    # If more than one face is detected
    elif len(detections) > 1:
        print(f'to many faces in file: {filename}')

    # If no face is detected
    elif len(detections) == 0:
        print(f'no faces faces in file: {filename}')

    else:
        print('check this case')

# Getting all photos from the directory
all_photos = [filename for filename in os.listdir(data_folder) if filename.endswith('.jpg')]

# Extracting face coordinates for all photos
for filename in all_photos:
    face_coordinats(filename)