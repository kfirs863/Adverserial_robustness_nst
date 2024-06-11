import os
from collections import OrderedDict

import torch
import torchmetrics
from PIL import Image as PIL_Image
from RealESRGAN import RealESRGAN
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from datasets import Dataset, load_dataset, Image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CometLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils import model_zoo


from custom_transformers.adversarial_attack_transformer import AdversarialAttack, set_requires_grad
from custom_transformers.real_esrgan_transform import RealESRGANTransform
from modules.adain import AdINStyleTransferBlock
from modules.resnet_sin import ResNetSIN
from utils import get_transform


# Function to create adversarial examples using FGSM
def create_adversarial_examples(model, images, epsilon=8 / 256):
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


# Evaluation Model with torchmetrics
class EvaluationModel(LightningModule):
    def __init__(self, model: nn.Module, num_classes=1000):
        super().__init__()
        self.upscaler_transforms = None
        self.model = model
        # self.adain = AdINStyleTransferBlock(device)

        # # Initialize the RealESRGAN model and load its weights
        # self.upscaler = RealESRGAN(device, scale=2)
        # self.upscaler.load_weights('weights/RealESRGAN_x2.pth', download=True)
        #
        # # Create the custom transform
        # self.upscaler_transform = RealESRGANTransform(self.upscaler)

        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # Define top-5 validation accuracy
        self.val_top5_accuracy = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=num_classes)
        self.test_top5_accuracy = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=num_classes)

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage):
        self.upscalers = []
        for i in range(torch.cuda.device_count()):
            upscaler = RealESRGAN(f'cuda:{i}', scale=2)
            upscaler.load_weights('weights/RealESRGAN_x2.pth', download=True)
            self.upscalers.append(upscaler)
        self.upscaler_transforms = [RealESRGANTransform(upscaler) for upscaler in self.upscalers]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.CrossEntropyLoss()(output, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x = self.adain.apply(content_image=x)
        # x = self.upscaler.predict(x)
        #
        gpu_index = self.device.index if isinstance(self.device, torch.device) else 0

        x = self.upscaler_transforms[gpu_index](x)
        # # downscale the image to 224x224
        x = transforms.Resize((224, 224))(x)

        x = x.to(self.device)
        output = self(x)
        y_pred = torch.argmax(output, dim=1)
        self.val_accuracy(y_pred, y)
        self.val_top5_accuracy(output, y)
        self.log('val_acc', self.val_accuracy, prog_bar=True, on_step=True, on_epoch=False)
        self.log('val_top5_acc', self.val_top5_accuracy, prog_bar=True, on_step=True, on_epoch=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = create_adversarial_examples(self.model, x)
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
        self.log('test_top5_acc_epoch', self.test_top5_accuracy, prog_bar=True, on_step=False, on_epoch=True)


if __name__ == '__main__':
    weights = models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    # transform = get_transform()

    # Load the dataset using the load_dataset method
    dataset: Dataset = load_dataset('imagenet-1k', cache_dir='/mobileye/RPT/users/kfirs/kfir_project',
                                    trust_remote_code=True, split='validation')
    dataset_folder = ImageFolder(root='/mobileye/RPT/users/kfirs/kfir_project/adversarials_images', transform=transform,target_transform=str)

    val_dataloader_folder = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    def transforms(examples):
        examples["image"] = [
            transform(image) for image in examples["image"]
        ]

        return examples

    # Apply the transform to the dataset
    dataset = dataset.cast_column("image", Image(mode="RGB"))
    dataset.set_format("torch")
    dataset.set_transform(transforms)


    # Define a custom collate function
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return images, labels


    val_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4,
                                collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cpu'

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, num_classes=1000)
    # Load pretrained model of ResNet on Stylized ImageNet (Resnet_SIN)
    # model = models.resnet50(weights=None)
    # url = 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar'
    #
    # checkpoint = model_zoo.load_url(url, map_location=torch.device('cpu'))

    # Remove prefix module. for all keys in checkpoint['state_dict']
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['state_dict'].items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    #
    # # Load the new state dict
    # model.load_state_dict(new_state_dict)

    # Set the environment variable for the Comet URL
    os.environ['COMET_URL_OVERRIDE'] = 'https://www.comet.com/clientlib/'

    # Define Comet ML Logger
    comet_logger = CometLogger(
        api_key="jsPqM9osr1ZfIKWiEeiAlitCa",
        project_name="adversarial-robustness-nts",
    )

    # Model and Trainer
    pl_model = EvaluationModel(model=model)
    trainer = Trainer(max_epochs=1, accelerator='auto' if device == 'cuda' else 'cpu', logger=comet_logger)

    # Test the model
    trainer.validate(pl_model, val_dataloader)
