import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL.JpegImagePlugin import JpegImageFile
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from datasets import load_dataset, Dataset, Image
import torch
from torchvision import models
from PIL import Image as PIL_Image
from torchvision.transforms import transforms
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image


PATH_TO_SAVE = '/mobileye/RPT/users/kfirs/kfir_project/adversarials_subset_images'

with open('imagenet-simple-labels.json') as f:
    labels = json.load(f)


def class_id_to_label(i):
    return labels[i]

# Function to transform images
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

# Function to create adversarial examples using FGSM
def create_adversarial_examples(model, images, epsilon=8/256):
    # Get the min and max pixel values of the dataset
    min_pixel_value, max_pixel_value = images[0].min(), images[0].max()

    # Create PyTorch classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(min_pixel_value, max_pixel_value),
        device_type='cpu'
    )

    # Create FGSM attack
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)

    # Generate adversarial test examples
    adversarial_images = attack.generate(x=images.numpy())
    return torch.from_numpy(adversarial_images)


# Step 1: Load the dataset
dataset: Dataset = load_dataset('imagenet-1k', cache_dir='/mobileye/RPT/users/kfirs/kfir_project',
                                trust_remote_code=True, split='validation[:10%]')

# Create a directory to save processed images
os.makedirs(PATH_TO_SAVE, exist_ok=True)

dataset = dataset.cast_column("image", Image(mode="RGB"))

# Step 2: Load the pre-trained model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
transform = models.ResNet50_Weights.DEFAULT.transforms()

# Define a function to apply your custom function to a dataset example
def process_example(example,custom_transform):
    image = example['image']

    # Convert image to numpy array
    transformed_image = custom_transform(image)

    # # PIL to numpy
    # transformed_image = np.array(transformed_image)

    # Add channel dimension
    # image_np = np.expand_dims(transformed_image, axis=0)

    # Convert image to PyTorch format
    # transformed_image = image_np.transpose((0, 3, 1, 2)).astype(np.float32)
    # transformed_image = transformed_image.transpose((2, 0, 1)).astype(np.float32)

    transformed_image_tensor = torch.stack([transformed_image])

    # Apply custom function
    processed_image_np = create_adversarial_examples(model, transformed_image_tensor)

    # Convert image back to numpy array
    # processed_image_np = processed_image_np.transpose((0, 2, 3, 1))

    # Remove added channel dimension
    # processed_image_np = np.squeeze(processed_image_np)

    example['image'] = processed_image_np
    return example


# def process_and_save_example(i_example):
#     i, example = i_example
#     label_dir = Path(PATH_TO_SAVE, class_id_to_label(example['label']))
#     path_to_save = label_dir / f'image_{i}.jpg'
#
#     if not path_to_save.exists():
#         processed_example = process_example(example)
#         processed_image = PIL_Image.fromarray(processed_example['image'])
#         # processed_image: JpegImageFile = example['image']
#
#         os.makedirs(label_dir, exist_ok=True)
#         processed_image.save(path_to_save)


# # Serial processing of the dataset
for i, example in enumerate(tqdm(dataset)):
    label_dir = Path(PATH_TO_SAVE, class_id_to_label(example['label']))
    path_to_save = label_dir / f'image_{i}.jpg'
    custom_transform = get_transform()
    if not path_to_save.exists():
        processed_example = process_example(example,custom_transform)
        # Convert the tensor to a PIL Image
        image_pil = to_pil_image(processed_example['image'][0])
        os.makedirs(label_dir, exist_ok=True)
        # Save the image
        image_pil.save(path_to_save)


# with Pool() as p:
#     list(tqdm(p.imap(process_and_save_example, enumerate(dataset)), total=len(dataset)))

print('Image processing completed!')
