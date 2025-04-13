import json
import numpy as np
from PIL import Image, ImageDraw
import os

# json_path = "/home/nil/manipulation/dataset/pick_plum_100/pt_json/rgb_line.json"

json_path = None

assert json_path is not None, "Please set the json_path variable to the path of your JSON file."

# Load JSON data
with open(json_path, 'r') as f:
    data = json.load(f)
    
for sample in data:
    img = Image.open(sample['image'])
    traj = sample['line']
    
    # Draw the trajectory on the image
    draw = ImageDraw.Draw(img)
    for i in range(len(traj) - 1):
        draw.line((traj[i][0], traj[i][1], traj[i + 1][0], traj[i + 1][1]), fill="red", width=5)
    
    # Replace "rgb" with "traj" in the image path
    new_image_path = sample['image'].replace("rgb", "line_visual")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
    
    # Save the modified image
    img.save(new_image_path)
    print("saved", new_image_path)