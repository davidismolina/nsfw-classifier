"""Simple example script to train an NSFW classifier.

The script assumes you have already downloaded images into ``image_data/``
using ``imagescraper.py``. Porn and sexy images are the positive class and
neutral images are the negative class.

We fine-tune a pretrained ResNet18 (a convolutional neural network) using the
Adam optimizer.
"""

import os
from PIL import ImageFile
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from torch.optim import Adam

DATA_DIR = "image_data"
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4


def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Dataset folder '{DATA_DIR}' not found. Run imagescraper.py first.")

    # Basic preprocessing: resize and normalize images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    # Remove any unreadable or corrupted images before training
    valid_samples = []
    skipped = 0
    for path, label in dataset.samples:
        try:
            img = dataset.loader(path)
            img.load()
            valid_samples.append((path, label))
        except Exception as e:
            print(f"Skipped unreadable image: {path} ({e})")
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} unreadable images")
        dataset.samples = valid_samples
        dataset.imgs = valid_samples
        dataset.targets = [s[1] for s in valid_samples]

    # 80/20 train/validation split
    num_train = int(0.8 * len(dataset))
    num_val = len(dataset) - num_train
    train_data, val_data = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load pretrained ResNet18 and replace final layer for two classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer (Adam) for fine-tuning
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()


    