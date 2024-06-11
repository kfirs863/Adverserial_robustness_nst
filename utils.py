# Function to load and preprocess an image
import torch
from PIL import Image
from RealESRGAN import RealESRGAN
from torchvision.transforms import transforms

from custom_transformers.real_esrgan_transform import RealESRGANTransform


def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path)
    image = transform(image)
    return image


# Function to transform images
def get_transform():
    # # Device setup
    # device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cpu')
    #
    # # Initialize the RealESRGAN model
    # model = RealESRGAN(device, scale=4)
    # model.load_weights('weights/RealESRGAN_x4.pth', download=True)
    #
    # # Create the custom transform
    # sr_transform = RealESRGANTransform(model)

    return transforms.Compose([
        # sr_transform,
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])