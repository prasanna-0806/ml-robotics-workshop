## Task 1: Image PreProcessing

This script performs image preprocessing using OpenCV.

Unzipping the Images: 
The script checks if the images folder exists.
If not, it extracts images from images.zip into the folder.

Preprocessing Workflow for Each Image:
Reads each image in the images folder.
Resizes the image to 128x128 pixels.
Normalizes the pixel values by dividing them by 255 (scales them to the range 0-1).
Applies a Gaussian blur with a kernel size of 5x5 to smooth the image.
Converts the blurred image back to uint8 format for saving.
Saves the processed image into a new folder called processed_images.

Error Handling:
Skips files that are not valid images or are unreadable.

Final Output:
Prints status updates for each step in the preprocessing pipeline.
Outputs all processed images in the processed_images folder.

![awmleer-I--YyrXUphc-unsplash](https://github.com/user-attachments/assets/e013b32a-734d-44d6-a8a7-a72b556ce44f)
![daniel-lloyd-blunk-fernandez-QkKKggRWlE8-unsplash](https://github.com/user-attachments/assets/a43de8cb-63d8-4bb7-bda3-1f9164fad1a4)
![gabriel-martin-FH3NzSqwOTU-unsplash](https://github.com/user-attachments/assets/a8b4272f-6d7a-4e72-add3-8c85259d6085)

## Task 2: Feature Extraction

This script focuses on feature extraction from images using OpenCV. It includes both edge detection (using the Canny method) and corner detection (using the Harris Corner Detection method). 

Unzipping Images: 
Checks if the images folder exists. If not, it extracts images.zip into this folder.
Skips the extraction step if the folder already exists.

Feature Extraction Workflow:
Reads each file in the images folder.
For each valid image:
Converts the image to grayscale.
Performs edge detection using the Canny method with thresholds 100 and 200.
Detects corners using the Harris Corner Detection method, dilates the corners for better visibility, and highlights the corners in red on the original image.
Saves the edge-detected and corner-highlighted images to the processed_features folder with appropriate filenames (edges_ and corners_ prefixes).

Error Handling: 
Skips unreadable files or non-image files with clear console messages.

Output:
Status updates during feature extraction.
Saves processed images (both edge-detected and corner-highlighted) into the processed_features folder.


![corners_awmleer-I--YyrXUphc-unsplash](https://github.com/user-attachments/assets/3c93d125-f08e-40db-8db1-bc0fc89c084d)
![corners_daniel-lloyd-blunk-fernandez-QkKKggRWlE8-unsplash](https://github.com/user-attachments/assets/f2ac891a-ea95-46cc-97da-4383cbe5309f)
![corners_gabriel-martin-FH3NzSqwOTU-unsplash](https://github.com/user-attachments/assets/04bc9a25-00e3-4328-a13e-632eb44e21be)

![edges_awmleer-I--YyrXUphc-unsplash](https://github.com/user-attachments/assets/8cb5e8d2-8460-4cc1-929d-7b44588a62a4)
![edges_daniel-lloyd-blunk-fernandez-QkKKggRWlE8-unsplash](https://github.com/user-attachments/assets/4aebf808-defa-4a38-b081-5fed74141ecd)
![edges_gabriel-martin-FH3NzSqwOTU-unsplash](https://github.com/user-attachments/assets/a3bef9cd-55d4-4387-83da-f1bad69c8e46)

## Task 3: Object Detection

This script performs basic object detection using the Haar Cascade method in OpenCV to detect faces in images. It focuses on extracting and drawing bounding boxes around detected faces in the images. 

Unzipping Images:
Extracts images from images.zip into a folder named images, unless the folder already exists.

Loading Haar Cascade:
Loads the pre-trained Haar Cascade XML file for frontal face detection (haarcascade_frontalface_default.xml) from OpenCV's data directory.

Face Detection Workflow:
Iterates through all image files in the images folder.
For each valid image:
Converts the image to grayscale for faster processing.
Detects faces using the Haar Cascade method with the following parameters:
scaleFactor=1.1: Scales the image at each step to detect faces of varying sizes.
minNeighbors=5: Minimum number of neighbors each rectangle should have to retain it.
minSize=(30, 30): Minimum size of the detected face.
Draws green rectangles around detected faces on the original image.
Saves the annotated images to the output folder.

Error Handling:
Skips unreadable or invalid files and prints a warning message.

Output:
Saves processed images (with detected faces highlighted) in the output folder.

![awmleer-I--YyrXUphc-unsplash (1)](https://github.com/user-attachments/assets/7add6086-db6d-483f-ac03-3431fef72cef)
![daniel-lloyd-blunk-fernandez-QkKKggRWlE8-unsplash (1)](https://github.com/user-attachments/assets/a4bf5946-add8-4f6a-8276-c763d0885047)
![gabriel-martin-FH3NzSqwOTU-unsplash (1)](https://github.com/user-attachments/assets/108538f1-8e87-4d56-b2ed-3c00964445e3)



