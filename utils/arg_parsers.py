import argparse

def get_train_args():
    """
    Parses and returns the command line arguments for training the flower classifier.

    :return: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Flower Classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models/checkpoint.pth', help='Path to save the checkpoint file')
    parser.add_argument('--arch', type=str, default='vgg19', help='Model architecture')

    return parser.parse_args()

def get_predict_args():
    """
    Parses and returns the command line arguments for predicting with the flower classifier.

    :return: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Flower Classifier')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    
    return parser.parse_args()

