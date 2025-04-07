import ssl
ssl._create_default_https_context = ssl._create_unverified_context # This is a workaround for SSL certificate verification issues
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr
from torchvision.models import ResNet18_Weights

# Load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

with open("imagenet_labels.txt", "r") as f:
    imagenet_labels = [line.strip() for line in f.readlines()]


def fgsm_attack(image, label, epsilon=0.03):
    """"Generates adversarial examples using the FGSM method."""
    image.requires_grad = True
    output = model(image)
    loss = nn.CrossEntropyLoss()(output, label)
    model.zero_grad()
    loss.backward()
    perturbed_image = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed_image, 0, 1)

def pgd_attack(image, label, eps=0.03, alpha=0.005, iters=10):
    """Generates adversarial examples using the PGD method."""
    ori_image = image.clone().detach()
    perturbed = image.clone().detach()
    for _ in range(iters):
        perturbed.requires_grad = True
        output = model(perturbed)
        loss = nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()
        perturbed = perturbed + alpha * perturbed.grad.sign()
        perturbation = torch.clamp(perturbed - ori_image, min=-eps, max=eps)
        perturbed = torch.clamp(ori_image + perturbation, 0, 1).detach()
    return perturbed

# Main function
def classify(image, attack):
    """Classifies an image and applies an adversarial attack if specified."""
    org_size = image.size
    img = transform(image).unsqueeze(0).to("cpu")

    # Get original prediction for adversarial attack
    with torch.no_grad():
        output = model(img)
    label = output.argmax(dim=1)

    # Apply attack
    if attack == "FGSM":
        adv_img = fgsm_attack(img.clone(), label)
    elif attack == "PGD":
        adv_img = pgd_attack(img.clone(), label)
    else: # No attack
        adv_img = img.clone()

    # Predict on adversarial image
    with torch.no_grad():
        output = model(adv_img)
    pred_label = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_label].item()

    # Return image and prediction
    return transforms.ToPILImage()(adv_img.squeeze().cpu()).resize(org_size), f"Prediction: {imagenet_labels[pred_label]} ({confidence:.2%})"

# Gradio UI
interface = gr.Interface(
    fn=classify,
    inputs=[
        gr.Image(type="pil"),
        gr.Radio(["None", "FGSM", "PGD"], label="Adversarial Attack Method")
    ],
    outputs=[
        gr.Image(type="pil", label="Transformed Image"),
        gr.Text(label="Model Prediction")
    ],
    title="Adversarial Attack Visualizer"
)

interface.launch()
