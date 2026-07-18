import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_data, get_splits, HAM10000Dataset, eval_transform
from src.model import build_model, get_device
from src.evaluate import evaluate

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model_v2.pth")
BATCH_SIZE = 32


def main():
    device = get_device()
    print("Device:", device)

    df, class_to_idx = load_data(data_dir=os.path.join("..", "data"))
    train_df, val_df, test_df = get_splits(df)
    class_names = list(class_to_idx.keys())
    num_classes = len(class_names)

    test_loader = DataLoader(
        HAM10000Dataset(test_df, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # load the trained model
    model = build_model(num_classes=num_classes, unfreeze_last_block=True).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    # evaluate on the untouched test set
    results = evaluate(model, test_loader, device, class_names)

    print("\n=== TEST SET RESULTS ===")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print("\nPer-class report:")
    print(results['report'])

    # confusion matrix
    cm = confusion_matrix(results['labels'], results['preds'])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()

    cm_path = os.path.join(CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"\nConfusion matrix saved to: {cm_path}")

    # specifically highlight melanoma recall since it's the highest-stakes class
    if "mel" in class_names:
        mel_idx = class_names.index("mel")
        mel_mask = np.array(results['labels']) == mel_idx
        mel_correct = np.array(results['preds'])[mel_mask] == mel_idx
        mel_recall = mel_correct.sum() / mel_mask.sum() if mel_mask.sum() > 0 else float('nan')
        print(f"\nMelanoma (mel) recall: {mel_recall:.4f} "
              f"({mel_correct.sum()}/{mel_mask.sum()} correctly identified)")


if __name__ == "__main__":
    main()