import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve


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
    plt.figure(figsize=(6, 4))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names, rotation=90)

    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()


def plot_roc_curve(true_labels, predictions, model_name):
    """
    
    """
    
    # compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # plot ROC curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, 
             tpr, 
             color='blue',
             label=f'ROC Curve (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], 
             [0, 1], 
             color='turquoise',
             linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.title(f'{model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")
    plt.show()


def plot_train_loss_accuracy(epochs, train_losses, train_accuracies, val_losses, val_accuracies):
    """
    
    """

    # set up subplots
    fig, (ax1, ax2) = plt.subplots(2)

    # set height
    plt.figure(figsize=(6, 4))

    x = [i for i in range(1, epochs + 1)]
    colors = ['blue', 'turquoise']

    # add plot title
    ax1.set(title = "Accuracy and Loss Comparison")

    # plot accuracy
    ax1.set(ylabel = "Accuracy")
    # customize axes values
    ax1.set_xticks([])
    ax1.set_ylim([0, 1.0])

    # plot
    for i, (model_name, train_accuracy) in enumerate(train_accuracies.items()):
        ax1.plot(x, 
                 train_accuracy, 
                 color=colors[i],
                 linestyle = '-', 
                 label=f'Train: {model_name}')
    
    for i, (model_name, val_accuracy) in enumerate(val_accuracies.items()):
        ax1.plot(x, 
                 val_accuracy, 
                 color=colors[i],
                 linestyle = '--', 
                 label=f'Val: {model_name}')
        
    ax1.legend(fontsize = "8", ncol = 2, loc="lower center")

    # plot loss
    ax2.set(ylabel = "Loss")
    # customize axes values
    ax2.set_xticks(x)
    ax2.set_ylim([0, 1.0])

    # plot
    for i, (model_name, train_loss) in enumerate(train_losses.items()):
        ax2.plot(x, 
                 train_loss, 
                 color=colors[i],
                 linestyle = '-', 
                 label=f'Train: {model_name}')
    
    for i, (model_name, val_loss) in enumerate(val_losses.items()):
        ax2.plot(x, 
                 val_loss,
                 color=colors[i],
                 linestyle = '--', 
                 label=f'Val: {model_name}')
    
    plt.show()