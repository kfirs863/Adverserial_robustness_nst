import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torchmetrics import ConfusionMatrix
from pytorch_lightning.loggers import CometLogger
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from load_pretrained_models import load_model

model_A = "resnet50_trained_on_SIN"
model_B = "resnet50_trained_on_SIN_and_IN"
model_C = "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN"


# Load the Caltech-101 dataset
def load_caltech101_data(data_path, batch_size=32):
    from torchvision.datasets import ImageFolder
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 as expected by ResNet-50
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = ImageFolder(root=data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


class ResNet50Model(pl.LightningModule):
    def __init__(self, num_classes):
        super(ResNet50Model, self).__init__()
        self.model = load_model(model_name=model_A)  # Load pretrained model On Stylized ImageNet

        # Replace the final fully connected layer
        self.model.fc = torch.nn.Linear(self.model.module.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class PretrainedModelInference(pl.LightningModule):
    def __init__(self, pretrained_model):
        super(PretrainedModelInference, self).__init__()
        self.model = pretrained_model
        self.confmat = ConfusionMatrix(num_classes=pretrained_model.model.fc.out_features, task="multiclass")

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.confmat(preds, y)
        accuracy = (preds == y).float().mean()
        self.log('test_accuracy', accuracy, logger=True)
        return preds, y

    def on_test_epoch_end(self):
        # Compute the confusion matrix
        cm = self.confmat.compute().cpu().numpy()
        fig = self.plot_confusion_matrix(cm)
        self.logger.experiment.log_figure(figure_name='Confusion Matrix', figure=fig)
        plt.close(fig)
        self.confmat.reset()

    def configure_optimizers(self):
        return None  # No optimizer needed for inference

    def plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        return fig


# Main function to run the inference
if __name__ == '__main__':
    os.environ["COMET_URL_OVERRIDE"] = "https://www.comet.com/clientlib/"

    # Define paths and parameters
    data_dir = './data'
    num_classes = 101

    # Load the dataset
    test_loader: DataLoader = load_caltech101_data(data_path='data/caltech-101')

    # Load the pretrained model
    pretrained_model = ResNet50Model(num_classes=len(test_loader.dataset.classes))

    # Instantiate the Lightning module for inference
    model = PretrainedModelInference(pretrained_model=pretrained_model)

    # Create a CometML logger
    comet_logger = CometLogger(
        api_key="jsPqM9osr1ZfIKWiEeiAlitCa",
        project_name="adversarial-robustness-nts"
    )

    # Create Trainer
    trainer = pl.Trainer(accelerator='cpu',logger=comet_logger)

    # Run inference and log metrics
    trainer.test(model, test_loader)
