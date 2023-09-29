import torch
import pickle
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.datasets.celeba import CelebA
from torch.utils.data.dataloader import DataLoader


class MyCelebA:
    def __init__(self, root, split, transform, target_attr):
        self.transform = transform
        self.target_attr = target_attr

        self.celeba = CelebA(root=root, split=split, target_type="attr", transform=transform)

        self.bias_idx = 20  # Gender Attribute.

        if target_attr == "Blond_Hair":
            self.target_idx = 9
            if split in ["train"]:
                save_path = Path(root) / "celeba" / "pickles" / "blonde"
                if save_path.is_dir():
                    print(f"use existing blonde indices from {save_path}")
                    self.indices = pickle.load(open(save_path / "indices.pkl", "rb"))
                else:
                    self.indices = self.build_blonde()
                    print(f"save blonde indices to {save_path}")
                    save_path.mkdir(parents=True, exist_ok=True)
                    pickle.dump(self.indices, open(save_path / f"indices.pkl", "wb"))
                self.attr = self.celeba.attr[self.indices]
            else:
                self.attr = self.celeba.attr
                self.indices = torch.arange(len(self.celeba))

        else:
            raise AttributeError

        self.targets = self.attr[:, self.target_idx]
        self.bias_targets = self.attr[:, self.bias_idx]

    def build_blonde(self):
        bias_targets = self.celeba.attr[:, self.bias_idx]
        targets = self.celeba.attr[:, self.target_idx]
        selects = torch.arange(len(self.celeba))[(bias_targets == 0) & (targets == 0)]
        non_selects = torch.arange(len(self.celeba))[~((bias_targets == 0) & (targets == 0))]
        np.random.seed(1)
        np.random.shuffle(selects)
        indices = torch.cat([selects[:2000], non_selects])
        return indices

    def __getitem__(self, index):
        img, _ = self.celeba.__getitem__(self.indices[index])
        target, bias = self.targets[index], self.bias_targets[index]

        return img, target, bias

    def __len__(self):
        return len(self.targets)


def get_celeba(root, split, target_attr, img_size, batch_size, num_workers=4):
    if split == "train":
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    dataset = MyCelebA(root=root, split=split, transform=transform, target_attr=target_attr)

    print(f"\nget_celeba - split: {split} size: {len(dataset)}")

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True if split == "train" else False, num_workers=num_workers)

    return dataset, dataloader
