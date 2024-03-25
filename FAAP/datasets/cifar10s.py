import torch
import random
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class CIFAR_10S:
    def __init__(self, root, split, transform, skew_ratio):
        self.transform = transform
        train_valid = split == "train"
        self.cifar10 = CIFAR10(root, train=train_valid, download=True)
        self.images = self.cifar10.data
        self.targets = np.array(self.cifar10.targets)
        self.bias_targets = np.zeros_like(self.targets)
        self.split = split

        if not train_valid:
            self.build_split()

        if split == "train":
            print("********")
            print("Skew ratio used %.2f" % skew_ratio)
            print("********")

            self.corrupt_dataset(skew_ratio)

        else:
            self.corrupt_test_dataset()

        self.targets, self.bias_targets = torch.from_numpy(self.targets).long(), torch.from_numpy(self.bias_targets).long()

    def build_split(self):
        indices = {i: [] for i in range(10)}
        size_per_class = 1000
        for idx, tar in enumerate(self.targets):
            indices[tar].append(idx)

        if self.split == "test":
            start = 0
            end = int(size_per_class * 0.9)

        if self.split == "valid":
            start = int(size_per_class * 0.9)
            end = size_per_class

        final_indices = []
        for ind_group in indices.values():
            final_indices.extend(ind_group[start:end])

        np.random.seed(1)
        np.random.shuffle(final_indices)

        self.images = self.images[final_indices]
        self.bias_targets = self.bias_targets[final_indices]
        self.targets = self.targets[final_indices]

    def rgb_to_grayscale(self, img):
        """Convert image to gray scale"""

        pil_img = Image.fromarray(img)
        pil_gray_img = pil_img.convert("L")
        np_gray_img = np.array(pil_gray_img, dtype=np.uint8)
        np_gray_img = np.dstack([np_gray_img, np_gray_img, np_gray_img])

        return np_gray_img

    def corrupt_test_dataset(self):
        self.images_gray = np.copy(self.images)
        self.bias_targets_gray = np.copy(self.bias_targets)
        self.targets_gray = np.copy(self.targets)

        for idx, img in enumerate(self.images_gray):
            self.images_gray[idx] = self.rgb_to_grayscale(img)
            self.bias_targets_gray[idx] = 1

        self.images = np.concatenate((self.images, self.images_gray), axis=0)
        self.bias_targets = np.concatenate((self.bias_targets, self.bias_targets_gray), axis=0)
        self.targets = np.concatenate((self.targets, self.targets_gray), axis=0)

    def corrupt_dataset(self, skew_level):
        gray_classes = [0, 2, 4, 6, 8]

        samples_by_class = {i: [] for i in range(10)}
        for idx, target in enumerate(self.targets):
            samples_by_class[target].append(idx)

        for class_idx in tqdm(range(10), ascii=True):
            class_samples = samples_by_class[class_idx]
            if class_idx in gray_classes:
                samples_skew_num = int(len(class_samples) * skew_level)
            else:
                samples_skew_num = int(len(class_samples) * (1 - skew_level))

            random.seed(1)
            samples_skew = random.sample(class_samples, samples_skew_num)
            for sample_idx in samples_skew:
                self.images[sample_idx] = self.rgb_to_grayscale(self.images[sample_idx])
                self.bias_targets[sample_idx] = 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        target = self.targets[index]
        bias = self.bias_targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target, bias


def get_cifar10s(root, split, img_size, batch_size, skew_ratio=0.95, num_workers=4):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    if split == "test" or split == "valid":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    dataset = CIFAR_10S(root=root, split=split, transform=transform, skew_ratio=skew_ratio)

    print(f"\nget_cifar10s - split: {split} size: {len(dataset)}")

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True if split == "train" else False, num_workers=num_workers)

    return dataset, dataloader
