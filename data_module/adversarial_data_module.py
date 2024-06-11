import pytorch_lightning as pl
from datasets import Dataset, Image
from torch.utils.data import DataLoader

from custom_datasets.adversarial_dataset import AdversarialDataset
from torchvision import transforms


class AdversarialDataModule(pl.LightningDataModule):
    def __init__(self, data: Dataset, attack, batch_size=32, transform=None):
        super().__init__()
        self.data = data
        self.attack = attack
        self.batch_size = batch_size
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()])

    def process_data(self, dataset: Dataset) -> Dataset:
        """
        Process the dataset by casting the image column to Image and setting the transform and format
        :param dataset:
        :return:
        """
        dataset = dataset.cast_column("image", Image(mode="RGB"))
        return dataset

    def train_dataloader(self) -> DataLoader:
        dataset = self.process_data(self.data['train'])
        ds = AdversarialDataset(dataset['image'], dataset['label'], self.attack, transform=self.transform)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        dataset = self.data['validation']
        ds = AdversarialDataset(dataset['image'], dataset['label'], self.attack, transform=self.transform,)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        dataset = self.process_data(self.data['test'])
        ds = AdversarialDataset(dataset['image'], dataset['label'], self.attack, transform=self.transform)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)
