"""Dataset class for team classification from cropped car images.
Expects folder-per-class structure: root_dir/<class_name>/*.jpg
"""
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class DriverDataset(Dataset):
    def __init__(self, root_dir, img_size=224, train=True):
        self.root_dir = root_dir
        self.classes = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[class_name]))

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])

    def class_counts(self):
        """Returns list of sample counts per class, in class_idx order - used for class-weighted loss."""
        counts = [0] * len(self.classes)
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label