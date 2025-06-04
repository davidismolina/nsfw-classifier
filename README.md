# Classifier Project

This repo contains:
- **imagescraper.py**: Python script to read `raw_data/*/*.txt` URL lists and download images into `image_data/`.
- **raw_data/**: `.txt` files of image URLs, organized by category (`neutral/`, `porn/`, `sexy/`).
- **image_data/**: downloaded images, separated by category (`urls_neutral/`, `urls_porn/`, `urls_sexy/`).
- (Future) training scripts and model code.

## How to use

1. Install dependencies:  
   ```bash
   pip install requests
