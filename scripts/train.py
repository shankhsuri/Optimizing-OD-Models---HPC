import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor

from data_preprocessing import ObjectDetectionDataset
from utils import create_ssd_annotation
from ssd import SSD  # Assuming you have an ssd.py file with the SSD model implementation


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    # Define dataset and transforms
    transform = Compose([Resize((args.image_size, args.image_size)), ToTensor()])
    dataset = ObjectDetectionDataset(args.data_dir, transform=transform, annotation_fn=create_ssd_annotation)

    # Split the dataset into training and validation sets
    train_size = int(args.split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = SSD(num_classes=args.num_classes)  # Initialize the SSD model
    model.to(device)

    # Set the loss function based on the user's choice
    if args.loss_function == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_function == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_function == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Invalid loss function: {args.loss_function}")

    # Set the optimizer based on the user's choice
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    # Training loop
    for epoch in range(args.num_epochs):
        start_time = time.time()

        # Train the model
        model.train()
        data_load_time = 0
        train_time = 0
        for images, targets in train_loader:
            data_load_end_time = time.time()
            data_load_time += data_load_end_time - start_time

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            train_time += time.time() - data_load_end_time
            start_time = time.time()

        # Validate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                output = model(images)
                loss = criterion(output, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch: {epoch+1}, Train Loss: {loss.item()}, Val Loss: {val_loss}, Data Load Time: {data_load_time}, Train Time: {train_time}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--image_size', type=int, default=300, help='Size of the input images (width and height)')
    parser.add_argument('--num_classes', type=int, default=21, help='Number of object classes including background')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio of training data to the total dataset size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='Loss function for training (CrossEntropyLoss, BCEWithLogitsLoss, MSELoss)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training (SGD, Adam, RMSprop, etc.)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    main(args)
