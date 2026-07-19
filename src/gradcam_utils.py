import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import eval_transform


def get_gradcam(model):
    """
    Build a GradCAM object targeting the last conv layer of MobileNetV2's
    feature extractor. This is the layer whose activations best represent
    high-level spatial features right before classification.
    """
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    return cam


def generate_gradcam_overlay(model, cam, image_path, target_class=None):
    """
    Run Grad-CAM on a single image.

    Args:
        model: the trained model (already in eval mode, on the right device)
        cam: a GradCAM object from get_gradcam()
        image_path: path to the image file
        target_class: int class index to explain. If None, uses the model's
                      own top prediction (standard behavior).

    Returns:
        original_rgb: original image as a float numpy array [0,1], resized to model input size
        overlay: the Grad-CAM heatmap overlaid on the original image
        predicted_class: the model's predicted class index
        confidence: softmax confidence for the predicted class
    """
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = eval_transform(pil_image).unsqueeze(0)  # add batch dim

    # get prediction + confidence first
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()

    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]  # first (only) image in batch

    # prepare original image resized to match model input size, normalized to [0,1] for overlay
    resized = pil_image.resize((160, 160))
    original_rgb = np.array(resized).astype(np.float32) / 255.0

    overlay = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)

    return original_rgb, overlay, predicted_class, confidence


def plot_gradcam_grid(model, cam, image_paths, class_names, true_labels=None, save_path=None):
    """
    Plot original + Grad-CAM overlay side by side for a list of images.
    Useful for eyeballing multiple examples at once (Day 7 sanity check).
    """
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, path in enumerate(image_paths):
        original, overlay, pred_class, confidence = generate_gradcam_overlay(model, cam, path)

        axes[i, 0].imshow(original)
        axes[i, 0].axis('off')
        title = f"Original"
        if true_labels is not None:
            title += f" (true: {class_names[true_labels[i]]})"
        axes[i, 0].set_title(title, fontsize=10)

        axes[i, 1].imshow(overlay)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(
            f"Grad-CAM (pred: {class_names[pred_class]}, {confidence:.2f})", fontsize=10
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved Grad-CAM grid to: {save_path}")
    plt.close(fig)