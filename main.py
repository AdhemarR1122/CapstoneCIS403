import cv2
import os

# Replace this line with the absolute path to your image
image_path = os.path.abspath('group_yearUp.jpg')


# Load the pre-trained face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Replace this line with the path to your image
image_path = 'group_yearUp.jpg'
img = cv2.imread(image_path)

# Check if the image is loaded successfully
if img is None:
    print("Error: Unable to load the image.")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output image
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
