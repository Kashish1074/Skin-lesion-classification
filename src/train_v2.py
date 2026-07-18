import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_data, get_splits, HAM10000Dataset, train_transform, eval_transform
from src.model import build_model, get_optimizer, get_device
from src.evaluate import compute_class_weights, get_criterion, get_weighted_sampler, evaluate

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
NUM_EPOCHS = 12
BATCH_SIZE = 32


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def val_loss_only(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = get_device()
    print("Device:", device)

    df, class_to_idx = load_data(data_dir=os.path.join("..", "data"))
    train_df, val_df, test_df = get_splits(df)
    class_names = list(class_to_idx.keys())
    num_classes = len(class_names)

    class_weights, _ = compute_class_weights(train_df, num_classes)

    # CHANGE FROM v1: use WeightedRandomSampler instead of shuffle=True
    # this oversamples rare classes on top of the already-weighted loss
    sampler = get_weighted_sampler(train_df, class_weights)

    train_loader = DataLoader(
        HAM10000Dataset(train_df, train_transform),
        batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        HAM10000Dataset(val_df, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = build_model(num_classes=num_classes, unfreeze_last_block=True).to(device)
    class_weights = class_weights.to(device)
    criterion = get_criterion(class_weights)
    optimizer = get_optimizer(model)

    train_losses, val_losses, val_macro_f1s = [], [], []
    best_macro_f1 = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = val_loss_only(model, val_loader, criterion, device)
        val_results = evaluate(model, val_loader, device, class_names)
        val_macro_f1 = val_results['macro_f1']

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_macro_f1s.append(val_macro_f1)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_macro_f1={val_macro_f1:.4f} | "
              f"val_balanced_acc={val_results['balanced_accuracy']:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            # NOTE: saved as v2 so v1 (from Day 5) isn't overwritten —
            # lets you compare both before deciding which to keep
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_v2.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved new best v2 checkpoint (val_macro_f1={val_macro_f1:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="train loss")
    axes[0].plot(val_losses, label="val loss")
    axes[0].legend()
    axes[0].set_title("Loss curves (v2 — weighted sampler)")

    axes[1].plot(val_macro_f1s, label="val macro-F1", color="green")
    axes[1].legend()
    axes[1].set_title("Validation macro-F1 (v2)")

    plt.tight_layout()
    plot_path = os.path.join(CHECKPOINT_DIR, "training_curves_v2.png")
    plt.savefig(plot_path)
    plt.show()

    print(f"\nBest val macro-F1 (v2): {best_macro_f1:.4f}")
    print(f"Compare this to your v1 result from Day 5 to see if the sampler helped.")


if __name__ == "__main__":
    main()