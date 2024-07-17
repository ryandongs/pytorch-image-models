import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import argparse

def main(args):
    # Load the model
    model = timm.create_model(
        'vit_base_patch16_224',
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint_path
    )

    # Set model to eval mode for inference
    model.eval()

    # Create Transform
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    print("Pretrained Config:", model.pretrained_cfg)

    # Manually define the labels for the custom dataset
    labels = args.labels.split(',')

    # Iterate through all images in the folder
    for image_name in os.listdir(args.image_folder):
        image_path = os.path.join(args.image_folder, image_name)
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Inference Script")
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the model.')
    parser.add_argument('--checkpoint_path' , metavar='DIR', required=True, help='Path to the model checkpoint.')
    parser.add_argument('--labels', type=str, required=True, help='Comma-separated list of labels for the classes.')

    args = parser.parse_args()
    main(args)
