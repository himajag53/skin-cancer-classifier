import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def generate_classification_report(true_labels, predictions, class_names, model_name):
    """
    
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
    
    """

    plt.figure(figsize=(10, 5))

    plt.plot(train_losses[0], label='Train Loss - ResNet18')
    plt.plot(val_losses[0], label='Val Loss - ResNet18')
    plt.plot(train_losses[1], label='Train Loss - UNet')
    plt.plot(val_losses[1], label='Val Loss - UNet')

    plt.title(f'Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()