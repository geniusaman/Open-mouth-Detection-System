import cv2
import numpy as np

# Load the Haar Cascade classifier for mouth detection
mouth_cascade = cv2.CascadeClassifier(r"D:\mY_ZoNe\My tasks\open mouth detection\haarcascade_mcs_mouth.xml")

# Check if the cascade classifier was loaded successfully
if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')

# Open a video capture object (using the default camera, in this case)
cap = cv2.VideoCapture(0)

# Downscale factor for the captured frames
ds_factor = 0.5

while True:
    flag = True  # A flag to indicate if the mouth is open or closed

    # Read a frame from the video capture
    ret, frame = cap.read()

    # Resize the frame to improve processing speed
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect mouths in the frame using the Haar Cascade classifier
    mouth_rects = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=11)

    # Loop over the detected mouth rectangles
    for (x, y, w, h) in mouth_rects:
        if h > 36:  # Assuming an open mouth if the height of the detected region is greater than a threshold
            cv2.putText(gray, "Mouth open", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
            cv2.imshow('Mouth Detector', gray)
            flag = False  # Set the flag to indicate that the mouth is open
        else:
            # Draw circles at the corners of the detected mouth region
            cv2.circle(frame, (int(x + 0.1 * w), y), 3, (0, 0, 255), 3)
            cv2.circle(frame, (int(x + 0.9 * w), y), 3, (0, 0, 255), 3)

        # Break after processing the first detected mouth rectangle
        break

    # If no mouth is detected, show the original frame
    if flag:
        cv2.imshow('Mouth Detector', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
