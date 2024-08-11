import json
import os
import sys

import numpy as np
import torch
import torchmetrics
from RealESRGAN import RealESRGAN
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from datasets import load_dataset, Dataset, Image
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CometLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


from custom_transformers.adversarial_attack_transformer import AdversarialAttack
from custom_transformers.real_esrgan_transform import RealESRGANTransform
from modules.adain import AdINStyleTransferBlock
from modules.resnet_sin import ResNetSIN

with open('imagenet-simple-labels.json') as f:
    labels = json.load(f)

def label_to_class_id(label):
    return labels.index(label)

# Function to transform images
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


# Evaluation Model with torchmetrics
class EvaluationModel(LightningModule):
    def __init__(self, model: nn.Module, num_classes=1000,mapping=None,scale=2,transform=None):
        super().__init__()
        self.upscaler_transforms = None
        self.model = model
        self.mapping = mapping
        self.scale = scale
        self.transform = transform
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # Define top-5 validation accuracy
        self.val_top5_accuracy = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=num_classes)
        self.test_top5_accuracy = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=num_classes)

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage):
        self.upscalers = []
        self.adain = []
        if self.device.type == 'cuda':
            for i in range(torch.cuda.device_count()):
                upscaler = RealESRGAN(f'cuda:{i}', scale=self.scale)
                upscaler.load_weights(f'weights/RealESRGAN_x{self.scale}.pth', download=True)
                self.upscalers.append(upscaler)

                adain = AdINStyleTransferBlock(device=f'cuda:{i}')
                self.adain.append(adain)
        else:
            upscaler = RealESRGAN('cpu', scale=self.scale)
            upscaler.load_weights(f'weights/RealESRGAN_x{self.scale}.pth', download=True)
            self.upscalers.append(upscaler)

            adain = AdINStyleTransferBlock(device='cpu')
            self.adain.append(adain)

        self.upscaler_transforms = [RealESRGANTransform(upscaler) for upscaler in self.upscalers]

    def _upscaler_transform(self, x,y):
        gpu_index = self.device.index if self.device.type == 'cuda' else 0
        x = self.upscaler_transforms[gpu_index](x)
        x = x.to(y.device)
        return x

    def _adain_transform(self, x):
        gpu_index = self.device.index if self.device.type == 'cuda' else 0
        x = self.adain[gpu_index].apply(x)
        return x

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor([label_to_class_id(self.mapping[int(label)]) for label in y]).to(self.device)

        # # Upscale the image
        x = self._upscaler_transform(x, y)

        # Apply style transfer to each image in the batch
        x = self._adain_transform(x)

        # Every 50 batches, log the first image in the batch
        if batch_idx % 100 == 0:
            self.logger.experiment.log_image(x[0].to('cpu'), name='validation_image', image_channels='first', step=batch_idx)

        # Transform the image
        x = self.transform(x)

        output = self(x)
        y_pred = torch.argmax(output, dim=1)
        self.val_accuracy(y_pred, y)
        self.val_top5_accuracy(output, y)
        self.log('val_acc', self.val_accuracy, prog_bar=True, on_step=True, on_epoch=False)
        self.log('val_top5_acc', self.val_top5_accuracy, prog_bar=True, on_step=True, on_epoch=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        y_pred = torch.argmax(output, dim=1)
        self.test_accuracy(y_pred, y)
        self.test_top5_accuracy(output, y)
        self.log('test_acc', self.val_accuracy, prog_bar=True, on_step=True, on_epoch=False)
        self.log('test_top5_acc', self.val_accuracy, prog_bar=True, on_step=True, on_epoch=False)

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc_epoch', self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_top5_acc_epoch', self.val_top5_accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log('test_acc_epoch', self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_top5_acc_epoch', self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True)


if __name__ == '__main__':
    from torchvision import models, transforms

    # Use transforms from pre-trained weights
    weights = models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    # transform = get_transform()

    # Set the environment variable for the Comet URL
    os.environ['COMET_URL_OVERRIDE'] = 'https://www.comet.com/clientlib/'
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

    # Load the dataset using the load_dataset method
    # dataset: Dataset = load_dataset('imagenet-1k', cache_dir='/mobileye/RPT/users/kfirs/kfir_project', trust_remote_code=True,split='validation')

    device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cpu'

    # model = ResNetSIN()
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, num_classes=1000)
    model.eval()

    # Load the dataset using the ImageFolder class
    dataset= ImageFolder(root='PATH TO SAVE THE ADVERSARIAL IMAGES',transform=transforms.ToTensor())

    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Define Comet ML Logger
    comet_logger = CometLogger(
        api_key="COMET_API_KEY",
        project_name="adversarial-robustness-nts",
        experiment_name="resnet50_adversarial_attack_super_resolution_x2_with_style_transfer_0.05_sand",
    )

    # Model and Trainer
    pl_model = EvaluationModel(model=model,mapping=list(dataset.class_to_idx),scale=2,transform=transform)
    trainer = Trainer(max_epochs=1, accelerator='auto' if device == 'cuda' else 'cpu',logger=comet_logger)

    # Test the model
    trainer.validate(pl_model, val_dataloader)

