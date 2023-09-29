import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.datasets import CIFAR10

from datasets.utils import TwoCropTransform, get_confusion_matrix


class BiasedCifar10S:
    def __init__(self, root, transform, split, skewed_ratio):
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
            print("Skewed ratio used %.2f" % skewed_ratio)
            print("********")

            self.corrupt_dataset(skewed_ratio)

        else:
            self.corrupt_test_dataset()

        self.targets, self.bias_targets = torch.from_numpy(self.targets).long(), torch.from_numpy(self.bias_targets).long()

        self.eye_tsr = self.get_eye_tsr()

        self.confusion_matrix_org, self.confusion_matrix, self.confusion_matrix_by = get_confusion_matrix(
            num_classes=10, targets=self.targets, biases=self.bias_targets
        )

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

        random.shuffle(final_indices)

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

            samples_skew = random.sample(class_samples, samples_skew_num)
            for sample_idx in samples_skew:
                self.images[sample_idx] = self.rgb_to_grayscale(self.images[sample_idx])
                self.bias_targets[sample_idx] = 1

    def get_target_distro(self, target):
        num_biases = len(torch.unique(self.bias_targets))
        target_distro = []
        for bias in range(num_biases):
            target_distro.append(torch.sum(torch.logical_and(self.targets == target, self.bias_targets == bias)))

        return target_distro

    def get_eye_tsr(self):
        num_targets = len(torch.unique(self.targets))
        num_biases = len(torch.unique(self.bias_targets))

        eye_tsr = torch.zeros((num_targets, num_biases))

        for target in range(num_targets):
            target_distro = self.get_target_distro(target)
            eye_tsr[target, np.argmin(target_distro)] = 1

        return eye_tsr

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        target = self.targets[index]
        bias = self.bias_targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target, bias, index


def get_cifar10s(root, split, num_workers=8, batch_size=128, aug=False, two_crop=False, ratio=0, skewed_ratio=0.95):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    if split == "test" or split == "valid":
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        if aug:
            train_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    if two_crop:
        train_transform = TwoCropTransform(train_transform)

    dataset = BiasedCifar10S(root, train_transform, split, skewed_ratio=skewed_ratio)

    def clip_max_ratio(score):
        upper_bd = score.min() * ratio
        return np.clip(score, None, upper_bd)

    if ratio != 0:
        weights = [1 / dataset.confusion_matrix_by[c, b] for c, b in zip(dataset.targets, dataset.bias_targets)]

        if ratio > 0:
            weights = clip_max_ratio(np.array(weights))

        sampler = data.WeightedRandomSampler(weights, len(weights), replacement=True)

    else:
        sampler = None

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if sampler is None and split == "train" else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=two_crop,
    )

    return dataloader
