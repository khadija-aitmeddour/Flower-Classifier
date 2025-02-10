import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from utils.data_utils import process_image
from utils.model_utils import create_custom_model, load_checkpoint
from utils.arg_parsers import get_predict_args


def predict(image_path, model):
    """
    Predicts the class probabilities for an input image using a trained model.

    :param image_path: Path to the image file.
    :param model: The trained model.
    :return: A dictionary mapping class names to their corresponding probabilities.
    """
    
    image = Image.open(image_path)
    image = process_image(image)

    logps = model.forward(image)
    ps = torch.exp(logps)
    
    probs, idx = ps.topk(5, dim=1)

    idx = idx.squeeze().tolist()
    probs = probs.squeeze().tolist()
    
    idx_to_class = {idx:k for k, idx in model.class_to_idx.items()}
    
    classes_idx = [idx_to_class[i] for i in idx]
    classes_names = [cat_to_name[c] for c in classes_idx] 

    predictions = dict(zip(classes_names, probs))
    
    return predictions


def show_results(image_path, model):
    """
    Displays the image along with the top class predictions and their probabilities.

    :param image_path: Path to the image file.
    :param model: The trained model.
    :return: None
    """
    
    image = Image.open(image_path)
    image.thumbnail((300,300))

    predictions = predict(image_path, model)

    probs = list(predictions.values())[::-1]
    classes = list(predictions.keys())[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={'width_ratios': [1, 2]})

    ax[0].imshow(image)
    ax[0].axis("off") 

    ax[1].barh(classes, probs, height=0.6, color="royalblue")
    ax[1].set_xlabel("Probability")
    ax[1].set_title("Top Predictions")

    ax[1].set_xticks(np.arange(0, 1.1, 0.2))

    plt.tight_layout()  
    plt.show()

if __name__ == '__main__':
    args = get_predict_args()

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    model = create_custom_model('vgg19')
    load_checkpoint(args.checkpoint, model)
    show_results(args.image, model)
