import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os

# Load from Hub 🔥
model = timm.create_model(
    'vit_base_patch16_224',
    num_classes=3,
    checkpoint_path='./output/train/20240717-061324-vit_base_patch16_224-224/model_best.pth.tar'
)

# Set model to eval mode for inference
model.eval()

# Create Transform
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
print("Pretrained Config:", model.pretrained_cfg)

# Manually define the labels for the custom dataset
labels = ['Clean_Water', 'Irrelevant', 'Polluted_Water']

# Define the folder containing images
image_folder = '../input/watercl/valid/Clean_Water/'

# Iterate through all images in the folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    # Open and process the image
    image = Image.open(image_path).convert('RGB')
    x = transform(image).unsqueeze(0)

    # Pass inputs to model forward function to get outputs
    out = model(x)

    # Apply softmax to get predicted probabilities for each class
    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    # Get the highest predicted class
    value, index = torch.max(probabilities, 0)

    # Prepare the highest prediction
    prediction = {"image": image_name, "label": labels[index], "score": value.item()}
    print(prediction)
