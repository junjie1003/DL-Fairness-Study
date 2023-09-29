import os
import PIL
import torch
import pickle
import numpy as np

from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader


class UTKFace:
    def __init__(self, root, split, transform, bias_attr, bias_rate=0.9):
        self.root = Path(root) / "images"
        filenames = np.array(os.listdir(self.root))
        np.random.seed(1)
        np.random.shuffle(filenames)
        num_files = len(filenames)
        num_train = int(num_files * 0.8)
        target_attr = "gender"

        self.transform = transform
        self.target_attr = target_attr
        self.bias_rate = bias_rate
        self.bias_attr = bias_attr
        self.train = split == "train"

        save_path = Path(root) / "pickles" / f"biased_utk_face-target_{target_attr}-bias_{bias_attr}-{bias_rate}"
        if save_path.is_dir():
            print(f"use existing biased_utk_face from {save_path}")
            data_split = "train" if self.train else "test"
            self.files, self.targets, self.bias_targets = pickle.load(open(save_path / f"{data_split}_dataset.pkl", "rb"))
            if split in ["valid", "test"]:
                save_path = Path(f"/root/study/data/clusters/utk_face_rand_indices_{bias_attr}.pkl")
                if not save_path.exists():
                    rand_indices = torch.randperm(len(self.targets))
                    pickle.dump(rand_indices, open(save_path, "wb"))
                else:
                    rand_indices = pickle.load(open(save_path, "rb"))
                num_total = len(rand_indices)
                num_valid = int(0.5 * num_total)

                if split == "valid":
                    indices = rand_indices[:num_valid]
                elif split == "test":
                    indices = rand_indices[num_valid:]

                indices = indices.numpy()

                self.files = self.files[indices]
                self.targets = self.targets[indices]
                self.bias_targets = self.bias_targets[indices]
        else:
            train_dataset = self.build(filenames[:num_train], train=True)
            test_dataset = self.build(filenames[num_train:], train=False)

            print(f"save biased_utk_face to {save_path}")
            save_path.mkdir(parents=True, exist_ok=True)
            pickle.dump(train_dataset, open(save_path / f"train_dataset.pkl", "wb"))
            pickle.dump(test_dataset, open(save_path / f"test_dataset.pkl", "wb"))

            self.files, self.targets, self.bias_targets = train_dataset if self.train else test_dataset

        self.targets, self.bias_targets = torch.from_numpy(self.targets).long(), torch.from_numpy(self.bias_targets).long()

    def build(self, filenames, train=False):
        attr_dict = {
            "age": (
                0,
                lambda x: x >= 20,
                lambda x: x <= 10,
            ),
            "gender": (1, lambda x: x == 0, lambda x: x == 1),
            "race": (2, lambda x: x == 0, lambda x: x != 0),
        }
        assert self.target_attr in attr_dict.keys()
        target_cls_idx, *target_filters = attr_dict[self.target_attr]
        bias_cls_idx, *bias_filters = attr_dict[self.bias_attr]

        target_classes = self.get_class_from_filename(filenames, target_cls_idx)
        bias_classes = self.get_class_from_filename(filenames, bias_cls_idx)

        total_files = []
        total_targets = []
        total_bias_targets = []

        for i in (0, 1):
            major_idx = np.where(target_filters[i](target_classes) & bias_filters[i](bias_classes))[0]
            minor_idx = np.where(target_filters[1 - i](target_classes) & bias_filters[i](bias_classes))[0]
            np.random.seed(1)
            np.random.shuffle(minor_idx)

            num_major = major_idx.shape[0]
            num_minor_org = minor_idx.shape[0]
            if train:
                num_minor = int(num_major * (1 - self.bias_rate))
            else:
                num_minor = minor_idx.shape[0]
            num_minor = min(num_minor, num_minor_org)
            num_total = num_major + num_minor

            majors = filenames[major_idx]
            minors = filenames[minor_idx][:num_minor]

            total_files.append(np.concatenate((majors, minors)))
            total_bias_targets.append(np.ones(num_total) * i)
            total_targets.append(np.concatenate((np.ones(num_major) * i, np.ones(num_minor) * (1 - i))))

        files = np.concatenate(total_files)
        targets = np.concatenate(total_targets)
        bias_targets = np.concatenate(total_bias_targets)
        return files, targets, bias_targets

    def get_class_from_filename(self, filenames, cls_idx):
        return np.array([int(fname.split("_")[cls_idx]) if len(fname.split("_")) == 4 else 10 for fname in filenames])

    def __getitem__(self, index):
        filename, target, bias = (self.files[index], int(self.targets[index]), int(self.bias_targets[index]))
        X = PIL.Image.open(os.path.join(self.root, filename))

        if self.transform is not None:
            X = self.transform(X)

        return X, target, bias

    def __len__(self):
        return len(self.files)


def get_utkface(root, split, bias_attr, img_size, batch_size, bias_rate=0.9, num_workers=4):
    size_dict = {64: 72, 128: 144, 224: 256}
    load_size = size_dict[img_size]

    if split == "train":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    else:
        transform = transforms.Compose(
            [
                transforms.Resize(load_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    dataset = UTKFace(root=root, split=split, transform=transform, bias_attr=bias_attr, bias_rate=bias_rate)

    print(f"\nget_utkface - split: {split} size: {len(dataset)}")

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True if split == "train" else False, num_workers=num_workers)

    return dataset, dataloader
