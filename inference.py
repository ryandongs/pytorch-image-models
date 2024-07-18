import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import argparse

def main(args):
    # Load from Hub 🔥
    model = timm.create_model(
        args.model,
        num_classes=3,
        checkpoint_path=args.checkpoint
    )

    # Set model to eval mode for inference
    model.eval()

    # Create Transform
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    # Manually define the labels for the custom dataset
    labels = ['Clean_Water', 'Irrelevant', 'Polluted_Water']

    # Define the folder containing images
    image_folder = args.image_folder

    # Iterate through all images in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        # Open and process the image
        if os.path.isfile(image_path):
            image = Image.open(image_path).convert('RGB')
            x = transform(image).unsqueeze(0)

            # Pass inputs to model forward function to get outputs
            out = model(x)

            # Apply softmax to get predicted probabilities for each class
            probabilities = torch.nn.functional.softmax(out[0], dim=0)

            # Get the highest predicted class
            value, index = torch.max(probabilities, 0)

            # Prepare the highest prediction
            prediction = {"label": labels[index], "score": value.item(), "image": image_name}
            print(prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification Inference')
    parser.add_argument('image_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')

    args = parser.parse_args()
    main(args)
