# Flower Classifier Project

## Overview
This project is an image classifier that identifies different types of flowers using a pre-trained PyTorch model. The model is fine-tuned to improve accuracy in flower classification.
## Directory Structure
```
flower_classifier/
│── flowers/              
│── models/              
│── scripts/              
│   ├── train.py          
│   ├── predict.py    
│── utils/                
│   ├── data_utils.py    
│   ├── model_utils.py    
│── cat_to_names.json             
│── README.md             
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/khadija-aitmeddour/Flower-Classifier.git
   cd flower_classifier
   ```
2. Install dependencies in the [environment.yml](environment.yml) file.

## Usage
### Training the Model
Run the following command to train the model:
```sh
python -m scripts.train --data_dir path/to/data --epochs 10 --lr 0.001
```

### Making Predictions
Run the following command to show the top 5 predictions for a given image:
```sh
python -m scripts.predict --image_path path/to/image.jpg --checkpoint path/to/checkpoint
```


## Dependencies
- Python 3.x
- PyTorch
- Torchvision
- Pillow
- Matplotlib
- NumPy
