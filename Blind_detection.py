import cv2
import matplotlib.pyplot as plt

# Load the pre-trained Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the image
img = cv2.imread('C:\\Users\\kolli\\Downloads\\WhatsApp Image 2023-06-29 at 3.24.17 PM.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect eyes in the grayscale image
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Check if eyes are detected
if len(eyes) > 0:
    print("Eyes detected!")
else:
    print("No eyes detected.")

# Display the image with rectangles around the detected eyes
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img_rgb)
plt.axis('off')  # Optional: Hide axis
plt.show()