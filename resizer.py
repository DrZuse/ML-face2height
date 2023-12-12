from PIL import Image
import numpy as np
import pandas as pd
import os

data_folder = './celebheights'
resized_images = './resized_images'

if not os.path.exists(resized_images):
    os.makedirs(resized_images)

# image file name, height_cm
h_data = pd.read_csv(data_folder + '/metadata.csv')

# image file name, facial_area
f_data = pd.read_csv(data_folder + '/detections.csv')

# Merge the two dataframes on 'thumbnail_name', keeping only the common rows
merged_df = pd.merge(h_data, f_data, on='thumbnail_name', how='inner')

# Keep only the 'thumbnail_name' and 'height_cm' columns
final_df = merged_df[['thumbnail_name', 'height_cm']]

# Save the DataFrame to a CSV file
final_df.to_csv(resized_images + '/final.csv', index=False)

for index, row in f_data.iterrows():
    y = row['y']
    height = row['h']
    x = row['x']
    width = row['w']
    thumbnail_name = row['thumbnail_name']

    img = Image.open(os.path.join(data_folder, row['thumbnail_name']))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Extract the region from the image array
    region_array = img_array[y:y+height, x:x+width]

    # Convert the region array back to an image
    region_image = Image.fromarray(region_array)

    old_size = region_image.size

    # calculate the new size maintaining the aspect ratio
    ratio = min(128.0 / old_size[0], 128.0 / old_size[1])
    new_size = tuple([int(x*ratio) for x in old_size])

    # resize the image
    resized_img = region_image.resize(new_size, Image.ANTIALIAS)

    # create a new image with a black background
    new_img = Image.new('RGB', (128, 128), 'black')
    new_img.paste(resized_img, ((128-new_size[0])//2, (128-new_size[1])//2))
    
    output_array = np.array(new_img)

    # save the resized image
    new_img.save(os.path.join(resized_images, thumbnail_name[:-4] + '.bmp'))