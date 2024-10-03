import numpy as np
from PIL import Image
import os
from mylabels import color2label
import tqdm
from torchvision import transforms

input_folder = '/home/cap6411.student1/CVsystem/assignment/hw8/cityscapes/val/label/'
output_folder = '/home/cap6411.student1/CVsystem/assignment/hw8/cityscapes/val/mylabel/'

os.makedirs(output_folder, exist_ok=True)

tolerance = 50

image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

data_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
    ])

for filename in tqdm.tqdm(image_files, desc="Processing images"):
    # Load the RGB label image
    rgb_image_path = os.path.join(input_folder, filename)
    rgb_image = Image.open(rgb_image_path).convert('RGB')
    #rgb_image = data_transforms(input_image) 
    rgb_array = np.array(rgb_image)

    # Create an empty array for the grayscale mask
    height, width, _ = rgb_array.shape
    grayscale_mask = np.zeros((height, width), dtype=np.uint8)

    for color, class_id in color2label.items():
        mask = np.all(np.abs(rgb_array - color) <= tolerance, axis=-1)
        grayscale_mask[mask] = class_id.trainId 
        
    gray_image = Image.fromarray(grayscale_mask)
    #gray_image = gray_image.convert('RGB')
    output_image_path = os.path.join(output_folder, filename)  # Keep the same filename
    gray_image.save(output_image_path)

        #print(f"Processed and saved: {filename}")

print("All images processed successfully!")

