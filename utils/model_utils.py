import torch
from torch import nn
from torchvision import models

def create_custom_model(model_name='vgg19', pretrained=True, freeze_features=True,n_outputs=102, hidden_units=512):
    """
    Creates a custom model based on a pre-trained model from torchvision.

    :param model_name: Name of the pre-trained model to use (default: 'vgg19').
    :param pretrained: Whether to use a pre-trained model (default: True).
    :param freeze_features: Whether to freeze the feature parameters (default: True).
    :param n_outputs: Number of output classes (default: 102).
    :param hidden_units: Number of hidden units in the classifier (default: 512).
    :return: The customized model.
    """
    
    model = getattr(models, model_name)(pretrained=pretrained)
    
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    n_inputs = model.classifier[0].in_features

    model.classifier = nn.Sequential(nn.Linear(n_inputs, hidden_units),
                          nn.ReLU(),
                          nn.Linear(hidden_units, n_outputs),
                          nn.LogSoftmax(dim=1))
    
    return model

def save_checkpoint(path, model, optimizer, criterion, epochs):
    """
    Saves the model checkpoint.

    :param path: Path to save the checkpoint.
    :param model: The model to save.
    :param optimizer: The optimizer used during training.
    :param criterion: The loss function used during training.
    :param epochs: Number of epochs the model was trained for.
    :return: None
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion': criterion,
        'class_to_idx': model.class_to_idx,
        'epochs': epochs
    }
    torch.save(checkpoint, path)
    print('Checkpoint saved successfully!')
    return  

def load_checkpoint(filepath, model):
    """
    Loads a model checkpoint.

    :param filepath: Path to the checkpoint file.
    :param model: The model to load the state into.
    :return: The epoch at which the checkpoint was saved.
    """
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    epoch = checkpoint['epoch']
    
    return epoch