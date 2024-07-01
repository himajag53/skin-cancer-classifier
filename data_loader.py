from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

TRAIN_PATH = 'images/train'
TEST_PATH = 'images/test'

def load_data(batch_size=32):
    """
    Load training and test datasets using ImageFolder and apply transformations.

    Args:
    - train_dir (str): Path to the training dataset directory.
    - test_dir (str): Path to the test dataset directory.
    - batch_size (int): Batch size for DataLoader (default: 32).

    Returns:
    - train_loader (DataLoader): DataLoader for training dataset.
    - val_loader (DataLoader): DataLoader for validation dataset.
    - test_loader (DataLoader): DataLoader for test dataset.
    """

    # define transformations for training and test data
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),          # specify images to 224x224
            transforms.RandomHorizontalFlip(),      # randomly flip images horizontally
            transforms.RandomRotation(10),          # randomly rotate images by up to 10 degrees
            transforms.ToTensor(),                  # convert to PyTorch tensors
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize with mean and std values
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),          # specify images to 224x224
            transforms.ToTensor(),                  # convert to PyTorch tensors
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize with mean and std values
        ])
    }

    # load datasets
    train_dataset = datasets.ImageFolder(
        root=TRAIN_PATH,
        transform=data_transforms['train'])

    test_dataset = datasets.ImageFolder(
        root=TEST_PATH,
        transform=data_transforms['test'])

    # split training dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # create data loaders for training, validation, and test datasets
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True)

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False)

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    
    # class verification
    print("Class Indices:", train_dataset.dataset.class_to_idx)
    print("Number of Training Samples:", len(train_dataset))
    print("Number of Validation Samples:", len(val_dataset))
    print("Number of Test Samples:", len(test_dataset))

    return train_loader, val_loader, test_loader, train_dataset.dataset.classes
