"""
Read PyTorch model from .pth.tar checkpoint.
"""
import os
import sys
from collections import OrderedDict
import torch
import torchvision
import torchvision.models
from torch.utils import model_zoo


def load_model(model_name):
    device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cpu')

    model_urls = {
        'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
        'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
        'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    }

    if "resnet50" in model_name:
        print("Using the ResNet50 architecture.")
        model = torchvision.models.resnet50(pretrained=False)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        checkpoint = model_zoo.load_url(model_urls[model_name], map_location=device)

        # Remove 'module.' prefix if present
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
    elif "vgg16" in model_name:
        print("Using the VGG-16 architecture.")
        filepath = "./vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar"
        assert os.path.exists(
            filepath), "Please download the VGG model yourself from the following link and save it locally: https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK (too large to be downloaded automatically like the other modules)"
        model = torchvision.models.vgg16(pretrained=False)
        if torch.cuda.device_count() > 1:
            model.features = torch.nn.DataParallel(model.features)
        model = model.to(device)
        checkpoint = torch.load(filepath, map_location=device)
    elif "alexnet" in model_name:
        print("Using the AlexNet architecture.")
        filepath = "./alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar"
        assert os.path.exists(
            filepath), "Please download the AlexNet model yourself from the following link and save it locally: https://drive.google.com/drive/u/0/folders/1GnxcR6HUyPfRWAmaXwuiMdAMKlL1shTn"
        model = torchvision.models.alexnet(pretrained=False)
        if torch.cuda.device_count() > 1:
            model.features = torch.nn.DataParallel(model.features)
        model = model.to(device)
        checkpoint = torch.load(filepath, map_location=device)
    else:
        raise ValueError("Unknown model architecture.")

    model.load_state_dict(new_state_dict)
    return model.to(device)
if __name__ == "__main__":

    # Abbreviations:
    # SIN = Stylized-ImageNet
    #  IN = normal, standard ImageNet

    model_A = "resnet50_trained_on_SIN"
    model_B = "resnet50_trained_on_SIN_and_IN"
    model_C = "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN"

    # Note: these two modules correspond to the ones reported in Figure 11.
    # Hyperparameters (learning rate etc.) were NOT optimised (this was
    # done in the rebuttal period with limited time), thus these
    # modules have lower performance than a typical model would have.
    # If peak performance is important to you, I suggest to train the model
    # yourself.
    model_D = "vgg16_trained_on_SIN"
    model_E = "alexnet_trained_on_SIN"

    model = load_model(model_A) # change to different model as desired
    print("Model download completed.")

    # sanity check: print state_dict
    try:
        for k, v in model.module.state_dict().items():
            print(k)
    except AttributeError:
        for k, v in model.state_dict().items():
            print(k)

    print("Model printing completed.")
