import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_data, get_splits
from src.model import build_model, get_device
from src.gradcam_utils import get_gradcam, generate_gradcam_overlay

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")  # match whichever you're using


def main():
    device = get_device()

    df, class_to_idx = load_data(data_dir=os.path.join("..", "data"))
    train_df, val_df, test_df = get_splits(df)
    class_names = list(class_to_idx.keys())
    num_classes = len(class_names)

    # grab the SAME nv sample used in test_gradcam.py (first nv row in test set)
    nv_idx = class_to_idx["nv"]
    nv_row = test_df[test_df['label'] == nv_idx].iloc[0]
    image_path = nv_row['path']
    print(f"Inspecting image: {image_path}")

    model = build_model(num_classes=num_classes, unfreeze_last_block=True).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    cam = get_gradcam(model)
    original, overlay, pred_class, confidence = generate_gradcam_overlay(model, cam, image_path)

    # load the image at FULL original resolution (not resized to 160x160)
    full_res = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(full_res)
    axes[0].set_title(f"Full resolution original\n({full_res.size[0]}x{full_res.size[1]})")
    axes[0].axis('off')

    axes[1].imshow(original)
    axes[1].set_title("Resized to model input (160x160)\n— what the model actually sees")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f"Grad-CAM overlay\n(pred: {class_names[pred_class]}, {confidence:.2f})")
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(CHECKPOINT_DIR, "nv_hair_artifact_check.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved comparison to: {save_path}")
    print("Open the full-resolution image and look at the same region the")
    print("secondary hotspot pointed to — check for hair, crease, or texture there.")


if __name__ == "__main__":
    main()