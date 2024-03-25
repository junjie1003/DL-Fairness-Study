import os

project_dir = "/root/DL-Fairness-Study"


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, transform=None, split="Train", sensitive="race", target="Attractive", seed=0, skew_ratio=1.0, labelwise=False):

        if name == "utkface":
            from data_handler.utkface import UTKFaceDataset

            root = f"{project_dir}/data/utkface/images"
            return UTKFaceDataset(root=root, split=split, transform=transform, sensitive_attr=sensitive, labelwise=labelwise)

        elif name == "celeba":
            from data_handler.celeba import CelebA

            root = f"{project_dir}/data"
            return CelebA(root=root, split=split, transform=transform, target_attr=target, labelwise=labelwise)

        elif name == "cifar10s":
            from data_handler.cifar10 import CIFAR_10S

            root = f"{project_dir}/data/cifar10s"
            return CIFAR_10S(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio, labelwise=labelwise)
