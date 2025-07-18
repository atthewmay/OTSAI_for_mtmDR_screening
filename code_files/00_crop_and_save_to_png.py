import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Input and output directories
input_dir = "Messidor-2"  # Replace with your images directory
output_dir = "cropped_data"  # Replace with your output directory
os.makedirs(output_dir, exist_ok=True)

# Border size for cropping
border_size = 10  # Adjust this as needed to avoid cropping too close to the retina

# Process each TIFF and JPEG image in the directory
i=0
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.tiff', '.tif', '.jpeg', '.jpg')):
        # Load image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold to create a mask for the retina
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # Threshold may need adjusting
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if "tif" in filename and len(contours)>1:
#             print(f"{filename} has {len(contours)} found")
#             plt.figure()
#             plt.imshow(thresh)
#             plt.show()

        # Find the largest contour, assuming it corresponds to the retina
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Expand the bounding box by adding a border
        x = max(x - border_size, 0)
        y = max(y - border_size, 0)
        w = min(w + border_size, img.shape[1] - x)
        h = min(h + border_size, img.shape[0] - y)
        
        # Crop the image
        cropped_img = img[y:y+h, x:x+w]
        
        # Convert to RGB format and save as PNG
        cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        png_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, png_filename)
        cropped_img_pil.save(output_path, "PNG")

        if i%20==1:
            print(f"Cropped and saved {filename} as {png_filename}, processed {i}")
        i+=1

