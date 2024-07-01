import torch

from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """
    Trains a deep learning model on the training dataset.

    Args:
        model (torch.nn.Module): Deep learning model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader providing batches of training data.
        criterion (torch.nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimization algorithm for updating model parameters.
        device (torch.device): Device (CPU or GPU) on which to perform computations.
        num_epochs (int, optional): Number of training epochs (default is 5).

    Returns:
        train_losses (list): List of training losses computed for each epoch.
        train_accuracies (list): List of training accuracies computed for each epoch.
    """
    
    model.to(device)
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):

        # set model to training mode
        model.train()

        # initialize
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(inputs)
            # compute loss
            loss = criterion(outputs, labels)
            # backpropagate loss
            loss.backward() 
            # update weights
            optimizer.step()

            # update training loss
            running_loss += loss.item() * inputs.size(0)

            # compute training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += torch.sum(predicted == labels).item()
            total_predictions += labels.size(0)

            progress_bar.set_postfix(loss=f'{running_loss / len(train_loader.dataset):.4f}')
        
        # average training loss for epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        epoch_accuracy = correct_predictions / total_predictions
        train_accuracies.append(epoch_accuracy)

    return train_losses, train_accuracies


def validate_model(model, val_loader, criterion, device):
    """
    Validates a deep learning model on the validation dataset.

    Args:
        model (torch.nn.Module): Deep learning model to be validated.
        val_loader (torch.utils.data.DataLoader): DataLoader providing batches of validation data.
        criterion (torch.nn.Module): Loss function used for validation.
        device (torch.device): Device (CPU or GPU) on which to perform computations.

    Returns:
        val_losses (list): List containing average validation loss for the epoch.

    """

    # set model to evaluation mode
    model.eval()
    running_val_loss = 0.0
    val_losses = []

    # disable gradient computation for validation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)

    # average validation loss for epoch
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    return val_losses


def test_model(model, test_loader, criterion, device):
    """
    Tests a deep learning model on the test dataset.

    Args:
        model (torch.nn.Module): Trained deep learning model to be tested.
        test_loader (torch.utils.data.DataLoader): DataLoader providing batches of test data.
        criterion (torch.nn.Module): Loss function used for testing.
        device (torch.device): Device (CPU or GPU) on which to perform computations.

    Returns:
        test_loss (float): Average test loss computed over the entire test dataset.
        accuracy (float): Accuracy of the model on the test dataset, computed as the ratio 
            of correctly predicted samples to the total number of samples.
        true_labels (list): List of true labels from the test dataset.
        predictions (list): List of predicted labels from the model on the test dataset.
    """

    # set model to evaluation mode
    model.eval()

    # initialize
    all_preds = []
    all_labels = []
    test_loss = 0.0
    predictions = []
    true_labels = []

    # disable gradient computation for testing
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing', leave=True):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # compute and update testing loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            # get predictions
            _, predicted = torch.max(outputs, 1) 

            # save predictions and true labels
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # average test loss
    test_loss /= len(test_loader.dataset)
    # compute accuracy 
    accuracy = accuracy_score(true_labels, predictions)

    # print test loss and accuracy
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2%}")

    return test_loss, accuracy, true_labels, predictions