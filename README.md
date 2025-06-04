# Classifier Project

This repo contains:
- **imagescraper.py**: Python script to read `raw_data/*/*.txt` URL lists and download images into `image_data/`.
- **raw_data/**: `.txt` files of image URLs, organized by category (`neutral/`, `porn/`, `sexy/`).
- **image_data/**: downloaded images, separated by category (`urls_neutral/`, `urls_porn/`, `urls_sexy/`).
- **train_nsfw_classifier.py**: example PyTorch training script that fine-tunes a pretrained **ResNet18** CNN to build a simple NSFW detector.

## How to use

1. Install dependencies:
   ```bash
   pip install requests torch torchvision
   ```
2. Download images:
   ```bash
   python imagescraper.py
   ```
3. Train the classifier:
   ```bash
   python train_nsfw_classifier.py
   ```
   The script expects `image_data/` to contain subfolders for `urls_porn`, `urls_sexy`, and `urls_neutral`. Porn and sexy images are treated as the positive (NSFW) class while neutral images are the negative class.

## How the training script works

For beginners, the process is:

1. **Load the dataset** – `torchvision.datasets.ImageFolder` reads the images from `image_data/` and labels each subfolder automatically. Both `urls_porn` and `urls_sexy` are considered NSFW while `urls_neutral` is not.
2. **Transforms** – images are resized to 224×224 pixels, converted to tensors and normalized so they are ready for the neural network.
3. **Pretrained CNN** – the script loads `ResNet18` from `torchvision.models`. This is a convolutional neural network already trained on ImageNet. We replace the last fully connected layer so it outputs two classes (NSFW vs. not).
4. **Optimizer** – training uses the **Adam** optimizer with a small learning rate to fine‑tune the network.
5. **Training loop** – the data is split into training and validation sets. The model trains for a few epochs and reports the validation accuracy after each one.
6. **Saving** – when training finishes, the weights are stored in `nsfw_classifier.pth`.

This setup provides a minimal example that you can build on as you experiment with different models or hyperparameters.
