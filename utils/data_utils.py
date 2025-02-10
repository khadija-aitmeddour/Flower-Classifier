import torch
from torchvision import transforms, datasets


def process_image(image):
    """
    Processes an image for use in a PyTorch model.

    :param image: The image to process.
    :return: The processed image tensor.
    """
    transform = get_test_transforms()
    image = transform(image)
    image = image.view(1, *image.shape)
    
    return image


def get_train_transforms():
    """
    Returns the transformations to be applied to the training dataset.

    :return: A composition of transformations for the training dataset.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ])
    
def get_test_transforms():
    """
    Returns the transformations to be applied to the validation and test datasets.

    :return: A composition of transformations for the validation and test datasets.
    """
    return transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def get_dataloaders(data_dir):
    """
    Returns the data loaders for the training, validation, and test datasets.

    :param data_dir: The directory containing the dataset.
    :return: A tuple containing the training, validation, and test data loaders.
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    trainset = datasets.ImageFolder(train_dir, transform=get_train_transforms())
    validset = datasets.ImageFolder(valid_dir, transform=get_test_transforms())
    testset = datasets.ImageFolder(test_dir, transform=get_test_transforms())
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)
    
    return trainloader, validloader, testloader