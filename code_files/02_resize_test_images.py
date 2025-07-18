from PIL import Image
import os

# Directory containing the images to resize
input_dir = "test_data/"  # Replace with the directory path

# Target size
target_size = (100, 100)

# Process each JPEG file in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)

        # Resize and save the new image
        img_resized = img.resize(target_size)
        output_path = os.path.join(input_dir, f"{os.path.splitext(filename)[0]}_small.jpg")
        img_resized.save(output_path, "JPEG")

        print(f"Saved resized image to {output_path}")

