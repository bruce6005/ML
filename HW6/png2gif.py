import os
from PIL import Image

# Base directory containing subfolders with PNG files
base_directory = "output4"

# Iterate through each folder in the base directory
for folder_name in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder_name)
    
    # Check if it's a folder
    if os.path.isdir(folder_path):
        # Get all PNG files sorted (assuming naming order matters)
        png_files = sorted([file for file in os.listdir(folder_path) if file.endswith(".png")])
        
        if not png_files:
            print(f"No PNG files found in {folder_path}, skipping.")
            continue
        
        # Load images from the folder
        images = [Image.open(os.path.join(folder_path, file)) for file in png_files]
        
        # Save as an animated GIF
        output_gif_path = os.path.join(folder_path, f"{folder_name}.gif")
        images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=True, duration=200, loop=0)
        
        print(f"GIF saved at {output_gif_path}")
