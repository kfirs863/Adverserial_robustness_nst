import torch
import torch.nn as nn
from pathlib import Path

from PIL import Image
from torchvision.transforms import Compose, transforms

from AdaIN_pytorch.AdaIN import AdaINNet
from AdaIN_pytorch.utils import transform, linear_histogram_matching


# Define the AdaIN style transfer block
class AdINStyleTransferBlock:

    def __init__(self, device='cpu',color_control=False):
        vgg = torch.load('./AdaIN_pytorch/vgg_normalized.pth')
        self.model = AdaINNet(vgg)
        self.model.decoder.load_state_dict(torch.load('./AdaIN_pytorch/decoder.pth'))
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        self.color_control = color_control
        self.transform: Compose = transform(512)

        # style_pth = Path('AdaIN_pytorch/images/texture/paper_texture.jpg')
        style_pth = Path('data/styles/4_sand.jpg')
        self.style_tensor = self.transform(Image.open(style_pth)).unsqueeze(0).to(device)


    def apply(self, content_images, alpha=0.05):
        styled_images = []
        for content_image in content_images:
            content_image = content_image.to(self.device)
            content_tensor = self.transform(transforms.ToPILImage()(content_image.squeeze(0))).unsqueeze(0).to(self.device)

            style_tensor = self.style_tensor
            if self.color_control:
                style_tensor = linear_histogram_matching(content_tensor, self.style_tensor)

            style_enc = self.model.encoder(style_tensor)
            content_enc = self.model.encoder(content_tensor)
            transfer_enc = self.adaptive_instance_normalization(content_enc, style_enc)

            mix_enc = alpha * transfer_enc + (1 - alpha) * content_enc
            styled_image = self.model.decoder(mix_enc)
            styled_images.append(styled_image.squeeze(0))

        return torch.stack(styled_images)

    # def adaptive_instance_normalization(content, style, eps=1e-5):
    #     mu_c = torch.mean(content, dim=[2, 3], keepdim=True)
    #     sigma_c = torch.std(content, dim=[2, 3], keepdim=True) + eps
    #     mu_s = torch.mean(style, dim=[2, 3], keepdim=True)
    #     sigma_s = torch.std(style, dim=[2, 3], keepdim=True) + eps
    #     normalized_content = (content - mu_c) / sigma_c
    #     return normalized_content * sigma_s + mu_s

    @staticmethod
    def adaptive_instance_normalization(x, y, eps=1e-5):
        """
        Adaptive Instance Normalization. Perform neural style transfer given content image x
        and style image y.

        Args:
            x (torch.FloatTensor): Content image tensor
            y (torch.FloatTensor): Style image tensor
            eps (float, default=1e-5): Small value to avoid zero division

        Return:
            output (torch.FloatTensor): AdaIN style transferred output
        """

        mu_x = torch.mean(x, dim=[2, 3])
        mu_y = torch.mean(y, dim=[2, 3])
        mu_x = mu_x.unsqueeze(-1).unsqueeze(-1)
        mu_y = mu_y.unsqueeze(-1).unsqueeze(-1)

        sigma_x = torch.std(x, dim=[2, 3])
        sigma_y = torch.std(y, dim=[2, 3])
        sigma_x = sigma_x.unsqueeze(-1).unsqueeze(-1) + eps
        sigma_y = sigma_y.unsqueeze(-1).unsqueeze(-1) + eps

        return (x - mu_x) / sigma_x * sigma_y + mu_y
