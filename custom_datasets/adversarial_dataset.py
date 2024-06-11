import torch
from torch.utils.data import Dataset


class AdversarialDataset(Dataset):
    def __init__(self, data, targets, attack, transform=None):
        self.data = data
        self.targets = targets
        self.attack = attack
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].unsqueeze(0).numpy()
        label = self.targets[idx].item()

        # Apply attack
        adversarial_image = self.attack.generate(x=image)

        if self.transform:
            adversarial_image = self.transform(torch.tensor(adversarial_image).squeeze())

        return adversarial_image, label
