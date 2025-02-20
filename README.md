# Conditional GAN for Image Generation

This repository contains a PyTorch implementation of a Conditional Generative Adversarial Network (CGAN) for generating images. The model is trained on a dataset with labeled images, allowing it to generate images conditioned on class labels.

## Features
- Uses a **Generator** and **Discriminator** model with label conditioning.
- Implements **Adversarial Loss** for effective training.
- Supports **batch processing** with a custom dataset transformation.

## Hyperparameters
The model is trained using the following hyperparameters:
```python
latent_size = 100
num_epochs = 5
batch_size = 128
learning_rate = 0.0002
image_size = 28
num_channels = 1
num_classes = 10
```

## Dataset & Preprocessing
The dataset undergoes preprocessing using:
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
```
This normalizes pixel values for better model performance.

## Model Architecture

### Generator
- Takes in **random noise** and **class labels**.
- Uses **embedding layers** to conditionally generate images based on labels.
- Implements **transposed convolution layers** for upsampling.

### Discriminator
- Takes in **real or generated images** along with **class labels**.
- Uses **convolution layers** to extract features.
- Outputs a probability score indicating whether the input is real or fake.

## Training
- The Generator is trained to **fool the Discriminator** by generating realistic images.
- The Discriminator is trained to **distinguish real images from fake ones**.
- The loss functions optimize both networks using **adversarial loss**.
- Loss values for both networks are recorded for analysis.

## Usage
To train the model, run the script with:
```bash
python train.py
```
Ensure that PyTorch and required dependencies are installed.

## Dependencies
Install dependencies using:
```bash
pip install torch torchvision numpy matplotlib
```

## Results
During training, the model generates images based on class labels. Sample results can be visualized to monitor progress.


