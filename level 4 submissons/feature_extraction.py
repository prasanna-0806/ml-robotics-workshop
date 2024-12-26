import cv2
import os
import numpy as np
import zipfile
zip_file = "images.zip"
output_folder = "images"
if not os.path.exists(output_folder):
    print("unzipping the images.zip file...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    print(f"images extracted to the folder: {output_folder}")
else:
    print("images folder already exists, s kipping unzip step")
processed_folder = "processed_features"
os.makedirs(processed_folder, exist_ok=True)
for image_name in os.listdir(output_folder):
    image_path = os.path.join(output_folder, image_name)
    if os.path.isfile(image_path):
        print(f"processing {image_name}...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"couldn't read {image_name}, skipping it")
            continue
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)  
        print(f"edge detection completed for {image_name}")
        corners = cv2.cornerHarris(gray_image, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)  
        print(f"corner detection completed for {image_name}")
        image_with_corners = image.copy()
        image_with_corners[corners > 0.01 * corners.max()] = [0, 0, 255] 
        print(f"corners highlighted for {image_name}")
        edge_image_path = os.path.join(processed_folder, f"edges_{image_name}")
        corner_image_path = os.path.join(processed_folder, f"corners_{image_name}")
        cv2.imwrite(edge_image_path, edges)
        cv2.imwrite(corner_image_path, image_with_corners)
        print(f"processed images saved: {edge_image_path} and {corner_image_path}\n")
    else:
        print(f"skipping {image_name} because it's not a valid file")
print("feature extraction complete!")
