import torch
from torch import nn, optim
import time
from utils.data_utils import get_dataloaders
from utils.model_utils import create_custom_model, save_checkpoint
from utils.arg_parsers import get_train_args


def train(model, epochs, criterion, optimizer, trainloader, validloader):
    """
    Trains the model using the training dataset and runs a validation pass at each epoch.

    :param model: The model to train.
    :param epochs: Number of epochs to train the model.
    :param criterion: The loss function.
    :param optimizer: The optimizer used for training.
    :param trainloader: DataLoader for the training dataset.
    :param validloader: DataLoader for the validation dataset.
    :return: None
    """
    
    print('\nTraining in progress...')
    print('---------------------------------')
    for e in range(epochs):
        model.train()

        running_train_loss = 0
        running_validation_loss = 0
        accuracy = 0
        n_train = len(trainloader)
        n_valid = len(validloader)

        epoch_start = time.time()

        for inputs, labels in trainloader:

            inputs, labels= inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            train_loss = criterion(logps, labels)
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()
        else:
            with torch.no_grad():
                model.eval()
                for inputs, labels in validloader:
                    inputs, labels= inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)
                    validation_loss = criterion(logps, labels)

                    running_validation_loss += validation_loss

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            avg_train_loss = running_train_loss/n_train
            avg_val_loss = running_validation_loss/n_valid
            avg_val_accuracy = (accuracy/n_valid) *100

            print(f'Epoch {e+1} completed in : {(time.time() - epoch_start):.2f}s')
            print('--------------------------')
            print(f'Training Loss : {avg_train_loss:.4f}')
            print(f'Validation Loss : {avg_val_loss:.4f}')
            print(f'Validation Accuracy : {avg_val_accuracy:.2f}%\n')
            print('---------------------------------')
            print('Training completed successfully!')


def calculate_test_accuracy(model, testloader):
    """
    Calculates the accuracy of the model on the test dataset.

    :param model: The trained model.
    :param testloader: DataLoader for the test dataset.
    :return: None
    """
        
    accuracy = 0
    with torch.no_grad():
        model.eval()

        for inputs, labels in testloader:
            inputs, labels= inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        avg_test_accuracy = (accuracy/len(testloader))*100
        print(f'accuracy : {avg_test_accuracy:.2f}%')
        return 
  

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_train_args()
    data_dir = args.data_dir

    trainloader, validloader, testloader = get_dataloaders(data_dir)

    model = create_custom_model(args.model).to(device)
    model.class_to_idx = trainloader.dataset.class_to_idx
    
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), args.lr)
    
    train(model, args.epochs, criterion, optimizer, trainloader, validloader)
    calculate_test_accuracy(model, testloader)
    save_checkpoint(model, optimizer, criterion, args.epochs, args.checkpoint)