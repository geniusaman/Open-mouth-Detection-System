import cv2
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk


# Load the Haar Cascade classifier for mouth detection
mouth_cascade = cv2.CascadeClassifier(r"haarcascade_mcs_mouth.xml")

# Check if the cascade classifier was loaded successfully
if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')

class MouthDetectorApp:
    def __init__(self, master, video_source=0):
        self.master = master
        master.title("Mouth Detector")

        self.canvas = Canvas(master)
        self.canvas.pack()

        # Open a video capture object
        self.cap = cv2.VideoCapture(video_source)

        # Downscale factor for the captured frames
        self.ds_factor = 0.5

        self.update()

    def update(self):
        flag = True  # A flag to indicate if the mouth is open or closed

        # Read a frame from the video capture
        ret, frame = self.cap.read()

        # Resize the frame to improve processing speed
        frame = cv2.resize(frame, None, fx=self.ds_factor, fy=self.ds_factor, interpolation=cv2.INTER_AREA)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect mouths in the frame using the Haar Cascade classifier
        mouth_rects = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=11)

        # Loop over the detected mouth rectangles
        for (x, y, w, h) in mouth_rects:
            if h > 36:  # Assuming an open mouth if the height of the detected region is greater than a threshold
                self.canvas.create_text(100, 100, anchor=tk.NW, text="Mouth is open", fill="red")
                flag = False  # Set the flag to indicate that the mouth is open
            else:
                # Draw circles at the corners of the detected mouth region
                self.canvas.create_oval(int(x + 0.1 * w), y, int(x + 0.1 * w) + 6, y + 6, fill="red")
                self.canvas.create_oval(int(x + 0.9 * w), y, int(x + 0.9 * w) + 6, y + 6, fill="red")
                

            # Break after processing the first detected mouth rectangle
            break

        # If no mouth is detected, show the original frame
        if flag:
            self.photo = self.convert_to_photo(frame)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.master.after(10, self.update)

    def convert_to_photo(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        return photo

if __name__ == "__main__":
    root = tk.Tk()
    app = MouthDetectorApp(root)
    root.mainloop()

# Release the video capture object and close all windows
app.cap.release()
cv2.destroyAllWindows()
