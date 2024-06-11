import torch
from torchvision import transforms
from PIL import Image

class RealESRGANTransform:
    def __init__(self, model):
        self.model = model

    def __call__(self, img):
        # The input image should be a PIL image
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            return self.model.predict(img)
        else:
            # If the input is a tensor of size N, apply the model to each image
            if len(img.shape) >= 4:
                return torch.stack([self._transform_single_image(i) for i in img])
            else:
                return self._transform_single_image(img)

    def _transform_single_image(self, img):
        # Convert the image Tensor to a PIL image
        img = transforms.ToPILImage()(img.squeeze(0))

        img = self.model.predict(img)

        # Convert the image back to a tensor
        img = transforms.ToTensor()(img)

        return img