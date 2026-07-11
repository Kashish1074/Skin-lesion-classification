import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

IMG_SIZE = 160

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert("RGB")
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# ... (transforms and HAM10000Dataset class stay the same) ...

def load_data(data_dir="data", metadata_file="HAM10000_metadata.csv"):
    df = pd.read_csv(os.path.join(data_dir, metadata_file))

    image_paths = {}
    for d in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        folder = os.path.join(data_dir, d)
        for fname in os.listdir(folder):
            image_id = fname.replace(".jpg", "")
            image_paths[image_id] = os.path.join(folder, fname)

    df['path'] = df['image_id'].map(image_paths)

    classes = sorted(df['dx'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    df['label'] = df['dx'].map(class_to_idx)

    # save relative to this file's location (src/), not the caller's cwd
    save_path = os.path.join(os.path.dirname(__file__), "class_to_idx.json")
    with open(save_path, "w") as f:
        json.dump(class_to_idx, f)

    return df, class_to_idx


def get_splits(df, seed=42):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=seed
    )
    return train_df, val_df, test_df