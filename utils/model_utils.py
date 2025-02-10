import torch
from torch import nn
from torchvision import models
from class_to_idx import class_to_idx

def create_custom_model(model_name='vgg19', pretrained=True, freeze_features=True,n_outputs=102, hidden_units=512):

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
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.class_to_idx = checkpoint['class_to_idx']
    model.class_to_idx = class_to_idx
    
    epoch = checkpoint['epoch']
    
    return epoch