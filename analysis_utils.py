import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix


def generate_classification_report(true_labels, predictions, class_names, model_name):
    """
    Generates and displays the classification report and confusion matrix for a classification model.

    Args:
        true_labels (array-like): True labels of the data.
        predictions (array-like): Predicted labels from the model.
        class_names (list): List of class names.
        model_name (str): Name of the model for which the report is generated.
    """
    
    # classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))

    # confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)

    # plot confusion matrix
    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, val_losses):
    """
    Plots training and validation losses all models in single graph.

    Args:
        train_losses (dict): Dictionary where keys are model names and values are lists
        of training losses across epochs.
        val_losses (dict): Dictionary where keys are model names and values are lists
        of validation losses across epochs.
    """
    plt.figure(figsize=(10, 6))

    for model_name, train_loss in train_losses.items():
        plt.plot(train_loss, label=f'Train Loss - {model_name}')

    for model_name, val_loss in val_losses.items():
        plt.plot(val_loss, label=f'Val Loss - {model_name}')

    plt.title(f'Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()


def plot_train_accuracies(train_accuracies):
    """
    Plots training accuracies for all models in single graph.
    
    Args:
        train_accuracies (dict): Dictionary where keys are model names and values are lists of training accuracies.
    """
    plt.figure(figsize=(10, 6))

    for model_name, train_accuracies in train_accuracies.items():
        epochs = range(1, len(train_accuracies) + 1)
        plt.plot(epochs, 
                 train_accuracies, 
                 marker='o', 
                 linestyle='-', 
                 label=model_name)

    plt.title('Training Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()