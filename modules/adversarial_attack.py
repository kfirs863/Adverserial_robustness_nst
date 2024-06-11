import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torch import Tensor


# Adversarial Attack Block with IBM ART
class AdversarialAttackBlock:
    def __init__(self, model, nb_classes, epsilon=0.1, device='cpu'):
        super(AdversarialAttackBlock, self).__init__()
        self.model = model
        self.epsilon = epsilon

        # Wrap the PyTorch model with ART's PyTorchClassifier
        self.art_classifier = PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            input_shape=(3, 224, 224),
            nb_classes=nb_classes,
            device_type=device,
        )


        # Initialize the adversarial attack method
        self.attack = FastGradientMethod(estimator=self.art_classifier, eps=epsilon)

    def apply(self, image: Tensor):
        # Enable gradients for the image
        image.requires_grad = True

        # Convert the image to NumPy after enabling gradients
        image_np = image.cpu().detach().numpy()
        adv_image_np = self.attack.generate(x=image_np)
        adv_image = torch.tensor(adv_image_np).to(image.device)

        return adv_image
