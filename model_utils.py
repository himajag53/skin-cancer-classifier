import torch

from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """
    
    """
    
    model.to(device)
    train_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_train_loss = 0.0
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
            running_train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=f'{running_train_loss / len(train_loader.dataset):.4f}')
        
        # average training loss for epoch
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

    return train_losses


def validate_model(model, val_loader, criterion, device):
    """
    
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


def test_model(model, test_loader, criterion, device, class_names):
    """
    
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