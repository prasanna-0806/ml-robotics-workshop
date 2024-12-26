import cv2
import os
import numpy as np
import zipfile
zip_file = "images.zip"
output_folder = "images"
if not os.path.exists(output_folder):
    print(f"unzipping {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    print(f"images extracted to {output_folder}")
else:
    print(f"images folder already exists at {output_folder}, skipping unzip")
processed_folder = "processed_images"
os.makedirs(processed_folder, exist_ok=True)
for image_name in os.listdir(output_folder):
    image_path = os.path.join(output_folder, image_name)
    if os.path.isfile(image_path):
        print(f"processing {image_name}...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"could not read {image_name} skipping")
            continue
        resized_image = cv2.resize(image, (128, 128))
        print(f"resized {image_name} to 128x128 pixels")
        normalized_image = resized_image / 255.0
        print(f"normalized pixel values for {image_name}")
        blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)
        print(f"applied Gaussian blur to {image_name}")
        final_image = (blurred_image * 255).astype(np.uint8)
        print(f"converted {image_name} back to uint8 format")
        processed_path = os.path.join(processed_folder, image_name)
        cv2.imwrite(processed_path, final_image)
        print(f"saved processed image: {processed_path}\n")
    else:
        print(f"Skipping {image_name} as it is not a file")
print("image preprocessing completed")
