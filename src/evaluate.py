import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score


def compute_class_weights(train_df, num_classes):
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    train_counts = train_df['label'].value_counts().sort_index()
    total = len(train_df)

    class_weights = total / (num_classes * train_counts.values)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights, train_counts


def get_weighted_sampler(train_df, class_weights):
    """Build a WeightedRandomSampler for oversampling rare classes.
    Use this OR shuffle=True in the DataLoader, not both.
    """
    sample_weights = class_weights[train_df['label'].values]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


def get_criterion(class_weights):
    """Weighted cross-entropy loss."""
    return nn.CrossEntropyLoss(weight=class_weights)


def evaluate(model, dataloader, device, class_names):
    """Run inference over a dataloader and return balanced accuracy,
    macro-F1, and a full per-class classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )

    return {
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'report': report,
        'preds': all_preds,
        'labels': all_labels,
    }