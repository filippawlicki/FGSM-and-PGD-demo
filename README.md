# Adversarial Attack Visualizer

This is a simple interactive demo built with **Gradio** and **PyTorch** that demonstrates two popular adversarial attack methods:

- **FGSM (Fast Gradient Sign Method)**
- **PGD (Projected Gradient Descent)**

Users can upload an image, choose an attack method, and see:
- The transformed (adversarial) image
- The model's prediction (label and confidence)

## How It Works

A pretrained **ResNet18** model is used (from `torchvision.models`). The app:
1. Preprocesses your uploaded image.
2. Runs a forward pass to get the model's prediction.
3. Applies the selected adversarial attack (or none).
4. Shows the modified image and new prediction.

![Example](example.png)
