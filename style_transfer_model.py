import os

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from combine_image_folder_dataset import CombinedImageFolder
from modules.adain import AdINStyleTransferBlock
from modules.adversarial_attack import AdversarialAttackBlock
from modules.resnet_sin import ResNetSIN
from modules.style_selection_cnn import StyleSelectionCNN


class StyleTransferModel(LightningModule):
    def __init__(self, vgg_path, decoder_path, num_styles=2, epsilon=0.1, num_classes=1000,device='cpu'):
        super(StyleTransferModel, self).__init__()
        self.style_selection_cnn = StyleSelectionCNN(num_styles)
        self.adversarial_attack_block = AdversarialAttackBlock(self.style_selection_cnn.model, epsilon)
        self.style_transfer_block = AdINStyleTransferBlock(vgg_path, decoder_path, device)
        self.resnet_sin = ResNetSIN(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()  # Define your loss function

    def forward(self, content_image, style_images, target):
        # Generate adversarial image
        adv_image = self.adversarial_attack_block(content_image, target)

        # Select style image
        style_idx = self.style_selection_cnn(adv_image).argmax(dim=1)
        style_image = style_images[style_idx]

        # Apply style transfer
        styled_image = self.style_transfer_block(adv_image, style_image)

        # Classify styled image
        classification = self.resnet_sin(styled_image)
        return classification

    def training_step(self, batch, batch_idx):
        content_image, target, style_images = batch
        classification = self(content_image, style_images, target)
        loss = self.criterion(classification, target)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.style_selection_cnn.parameters(), lr=0.001)
        return optimizer


if __name__ == '__main__':
    # Set random seed for reproducibility
    seed_everything(42, workers=True)

    # Use transforms from pre-trained weights
    weights = models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()

    # insert AdversarialAttackTransform transformer into the pipeline
    transform



    device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cpu'

    # Paths to the content and style directories
    content_root = 'data/caltech-101'
    style_root = 'data/styles'

    # Assert that the content and style directories exist
    assert os.path.isdir(content_root), f'Content directory not found: {content_root}'
    assert os.path.isdir(style_root), f'Style directory not found: {style_root}'

    # Dataset and DataLoader
    dataset = CombinedImageFolder(content_root, style_root, transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Paths to the pre-trained VGG and decoder models
    vgg_path = './AdaIN_pytorch/vgg_normalized.pth'
    decoder_path = './AdaIN_pytorch/decoder.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Comet ML Logger
    comet_logger = CometLogger(
        api_key="jsPqM9osr1ZfIKWiEeiAlitCa",
        project_name="adversarial-robustness-nts",
    )
    os.environ["COMET_URL_OVERRIDE"] = "https://www.comet.com/clientlib/"


    # Model and Trainer
    model = StyleTransferModel(vgg_path, decoder_path, num_styles=dataset.style_len, epsilon=0.1, device=device,num_classes=len(dataset.content_dataset.classes))
    trainer = Trainer(max_epochs=10, logger=comet_logger, gpus=1 if device == 'cuda' else 0,fast_dev_run=True)
