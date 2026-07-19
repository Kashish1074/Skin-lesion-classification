import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_data, get_splits
from src.model import build_model, get_device
from src.gradcam_utils import get_gradcam, plot_gradcam_grid

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# CHANGE THIS to whichever checkpoint you decided to keep from Day 6
# (best_model.pth for v1, or best_model_v2.pth if the sampler variant won)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")


def main():
    device = get_device()

    df, class_to_idx = load_data(data_dir=os.path.join("..", "data"))
    train_df, val_df, test_df = get_splits(df)
    class_names = list(class_to_idx.keys())
    num_classes = len(class_names)

    model = build_model(num_classes=num_classes, unfreeze_last_block=True).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    cam = get_gradcam(model)

    # pick one sample image per class from the test set
    sample_paths = []
    sample_labels = []
    for cls_name in class_names:
        cls_idx = class_to_idx[cls_name]
        subset = test_df[test_df['label'] == cls_idx]
        if len(subset) > 0:
            sample_paths.append(subset.iloc[0]['path'])
            sample_labels.append(cls_idx)

    save_path = os.path.join(CHECKPOINT_DIR, "gradcam_examples.png")
    plot_gradcam_grid(model, cam, sample_paths, class_names, true_labels=sample_labels, save_path=save_path)

    print("\nDone. Open the saved image to inspect the heatmaps:")
    print(save_path)
    print("\nCheck: does the heatmap focus on the lesion itself, or on background")
    print("artifacts like hair, rulers, or skin markings? Note anything unusual —")
    print("this matters for your README's limitations section.")


if __name__ == "__main__":
    main()