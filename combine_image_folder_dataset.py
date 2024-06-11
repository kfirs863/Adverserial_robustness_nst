import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import resnet50
from PIL import Image

# Define the dataset class using ImageFolder for content images
class CombinedImageFolder(Dataset):
    def __init__(self, content_root, style_root, transform=None):
        self.content_dataset = datasets.ImageFolder(content_root, transform=transform)
        self.style_images, self.style_labels = self.load_style_images(style_root, transform)
        self.content_len = len(self.content_dataset)
        self.style_len = len(self.style_images)
        self.transform = transform

    def load_style_images(self, style_root, transform):
        style_images = []
        style_labels = []
        for filename in os.listdir(style_root):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                style_image = Image.open(os.path.join(style_root, filename)).convert('RGB')
                if transform:
                    style_image = transform(style_image)
                style_images.append(style_image)
                # Extract the style label from the filename prefix
                style_label = int(filename.split('_')[0])
                style_labels.append(style_label)
        return style_images, style_labels

    def __len__(self):
        return self.content_len

    def __getitem__(self, idx):
        content_image, content_label = self.content_dataset[idx]
        return content_image, content_label, self.style_images

