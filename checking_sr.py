import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = '/homes/kfirs/PycharmProjects/adversarial_robustness_nst/data/IMG_20210109_181714 Cropped.jpg'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('/homes/kfirs/PycharmProjects/adversarial_robustness_nst/data/sr_image.png')