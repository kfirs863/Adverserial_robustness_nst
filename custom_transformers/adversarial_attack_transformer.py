import torch
from torchvision.utils import _log_api_usage_once


# Utility function to set requires_grad
def set_requires_grad(tensor, requires_grad=True):
    tensor.requires_grad = requires_grad
    return tensor


class AdversarialAttack(torch.nn.Module):

    def __init__(self, attack):
        super().__init__()
        # _log_api_usage_once(self)
        self.attack = attack

    def forward(self, img):
        img = img.unsqueeze(0)  # Add an extra dimension for the batch size
        img = set_requires_grad(img, True)  # Ensure the tensor requires gradients
        print("Before attack.generate:")
        print(f"img.requires_grad: {img.requires_grad}")
        print(f"img.grad_fn: {img.grad_fn}")

        adv_img = self.attack.generate(x=img.detach().numpy())

        adv_img = torch.from_numpy(adv_img).squeeze(0).to(
            img.device)  # Remove the added dimension and move to the original device

        # Since adv_img is created by an external library, re-enable requires_grad
        adv_img = set_requires_grad(adv_img, False)  # Set to False after adversarial attack generation

        print("After attack.generate:")
        print(f"adv_img.requires_grad: {adv_img.requires_grad}")
        print(f"adv_img.grad_fn: {adv_img.grad_fn}")

        return adv_img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attack={self.attack})"
