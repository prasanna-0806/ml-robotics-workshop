import cv2
import os
import zipfile
zip_file_path = 'images.zip'
extracted_folder = 'images'
output_folder = 'output'
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(extracted_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for img_file in os.listdir(extracted_folder):
    img_path = os.path.join(extracted_folder, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"could not load {img_file} skipping...")
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    output_path = os.path.join(output_folder, img_file)
    cv2.imwrite(output_path, image)
    print(f"{img_file}: Detected {len(faces)} face(s) saved to {output_path}")
print(f"processing c omplete. check the '{output_folder}' folder ")
