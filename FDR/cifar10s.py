import sys
import time
import random
import argparse
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from numpy import mean, std

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset
from torchvision.datasets.vision import VisionDataset

from utils import *
from models import MyResNet

sys.path.insert(1, "/root/study")
from helper import set_seed
from metrics import get_metric_index, get_all_metrics, print_all_metrics

parser = argparse.ArgumentParser(description="fairness")
parser.add_argument("--method", type=str, default="M2")
parser.add_argument("--ft_epoch", type=int, default=1000)
parser.add_argument("--ft_lr", type=float, default=1e-3)
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--constraint", type=str, default="EO")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--data_path", type=str, default="../data/cifar10s")
parser.add_argument("--batch_size", type=int, default=-1)
parser.add_argument("--checkpoint", type=str, default=None)
args = parser.parse_args()

set_seed(args.seed)
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

result_dir = "../results/cifar10s"
result_path = Path(result_dir)
result_path.mkdir(parents=True, exist_ok=True)
fout = open("/".join([str(result_path), f"fdr_{args.constraint.lower()}.txt"]), "w")

results = {}
metric_index = get_metric_index()
for m_index in metric_index:
    results[m_index] = []

repeat_time = 10

######################
# Data Preprocessing #
######################


class CIFAR_10S(VisionDataset):
    def __init__(self, root, split, transform, skewed_ratio):
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

        self.targets, self.bias_targets = (
            torch.from_numpy(self.targets).long(),
            torch.from_numpy(self.bias_targets).long(),
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        target = self.targets[index]
        bias = self.bias_targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target, bias


def get_cifar10s(root, split, skewed_ratio=0.95, batch_size=128, num_workers=4):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    if split == "test" or split == "valid":
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
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

    dataset = CIFAR_10S(root, split, transform=train_transform, skewed_ratio=skewed_ratio)

    data_size = len(dataset)
    print(f"{split.capitalize()} Size: {data_size}")

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if split == "train" else False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return data_loader, data_size


#########################
# CIFAR-10S dataset
skewed_ratio = 0.95
root = args.data_path

train_loader, train_size = get_cifar10s(root, split="train")

val_loader, val_size = get_cifar10s(root, split="valid")

test_loader, test_size = get_cifar10s(root, split="test")


def train_per_epoch(model, optimizer, criterion, epoch, num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx, (images, labels, biases) in enumerate(train_loader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # move to GPU
        images, labels = images.to(device), labels.to(device)
        # if batch_idx == 0:
        #     print(images.shape, labels.shape)

        # forward
        outputs = model.forward(images)

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels).item()

    epoch_loss /= len(train_loader)
    epoch_acc /= train_size

    print("TRAINING Epoch %d/%d Loss %.4f Accuracy %.4f" % (epoch, num_epochs, epoch_loss, epoch_acc))


def valid_per_epoch(model, epoch, num_epochs, criterion):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx, (images, labels, biases) in enumerate(val_loader):
        # move to GPU
        images, labels = images.to(device), labels.to(device)

        # forward
        outputs = model.forward(images)

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)

        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels).item()

    epoch_loss /= len(val_loader)
    epoch_acc /= val_size

    print("VALID Epoch %d/%d Loss %.4f Accuracy %.4f" % (epoch, num_epochs, epoch_loss, epoch_acc))

    return epoch_loss


def Finetune(model, criterion, train_loader, val_loader, test_loader):
    model.eval()

    best_clf = None
    method = args.method

    ###################################################
    # Option 1 (B1): Directly test
    # Option 2 (B2): Finetune on validation dataset
    # Option 3 (B3): Finetune on balanced-sampled (training dataset + validation dataset)
    # Option 4 (M1): Finetune on validation dataset + constraint
    # Option 5 (M2): Finetune on balanced-sampled + constraint
    ###################################################

    ################
    # Prepare the dataset
    ################
    x_train, y_train, a_train = prepare_cifar10s_data(train_loader, model, device)
    print(x_train.shape, y_train.shape, a_train.shape)

    x_test, y_test, a_test = prepare_cifar10s_data(test_loader, model, device)
    print(x_test.shape, y_test.shape, a_test.shape)

    x_finetune, y_finetune, a_finetune = prepare_cifar10s_data(val_loader, model, device)
    print(x_finetune.shape, y_finetune.shape, a_finetune.shape)

    if method == "B1":
        x_finetune = x_train
        y_finetune = y_train
        a_finetune = a_train

    elif method == "B2" or method == "M1":
        pass

    elif method == "B3" or method == "M2":  # Sample a balanced dataset
        X = torch.cat([x_train, x_finetune])
        Y = torch.cat([y_train, y_finetune])
        A = torch.cat([a_train, a_finetune])
        g_idx = []

        # i = 0, 1
        # j = 0, 1, ..., 9
        for i in range(2):  # Sensitive attribute
            for j in range(10):  # Target attribute
                g_idx.append(torch.where((A == i) & (Y == j))[0])  # (i,j)

        # 对于每个组合标签的索引列表，打乱数据点的顺序
        for i, g in enumerate(g_idx):
            idx = torch.randperm(g.shape[0])
            g_idx[i] = g[idx]

        # 计算所有组合标签中数据点数量最少的数量
        min_g = min([len(g) for g in g_idx])
        print(min_g)

        # 根据最小数量截取数据，以创建平衡的子数据集
        temp_g = torch.cat([g[:min_g] for g in g_idx])
        x_finetune = X[temp_g]
        y_finetune = Y[temp_g]
        a_finetune = A[temp_g]

    #############
    # Fine-tune #
    #############
    model.train()
    model.set_grad(False)
    model.append_last_layer(num_classes=10)
    model = model.to(device)
    # optimizer = optim.SGD(model.out_fc.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.NAdam(model.out_fc.parameters(), lr=args.ft_lr, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epoch)
    finetune_dataset = TensorDataset(x_finetune, y_finetune, a_finetune)
    # For B3 and M2, considering balance, only tried full batch
    if args.batch_size < 0:
        batch_size = y_finetune.shape[0]
    else:
        batch_size = args.batch_size
    print(batch_size)
    finetuneloader = torch.utils.data.DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True)
    print(len(finetune_dataset))

    weights = max(torch.bincount(y_finetune)) / torch.bincount(y_finetune)
    class_weights = torch.FloatTensor(weights).to(device)
    print(torch.bincount(y_finetune))
    print(class_weights)

    losses = []
    trigger_times = 0
    best_loss = 1e9
    for epoch in range(1, args.ft_epoch + 1):
        epoch_loss = 0.0
        epoch_loss_fairness = 0.0
        epoch_acc = 0.0
        for batch_idx, (x, y, a) in enumerate(finetuneloader):
            x, y, a = x.to(device), y.to(device), a.to(device)
            optimizer.zero_grad()
            outputs = model.out_fc(x)
            log_softmax, softmax = F.log_softmax(outputs, dim=1), F.softmax(outputs, dim=1)
            if method == "M1" or method == "M2":  # Use the fairness constraint
                if args.constraint == "MMF":
                    loss = mmf_constraint_cifar10s(criterion, log_softmax, y, a)
                else:
                    if args.constraint == "EO":
                        fpr, fnr = eo_constraint_cifar10s(softmax[:, 1], y, a)
                        loss_fairness = fpr + fnr
                    # elif args.constraint == 'DP':
                    #     loss_fairness = dp_constraint(softmax[:, 1], a)
                    elif args.constraint == "DI":
                        loss_fairness = di_constraint(softmax[:, 1], a)
                    elif args.constraint == "AE":
                        loss_fairness = ae_constraint(criterion, log_softmax, y, a)
                    epoch_loss_fairness += loss_fairness.item()
                    loss_1 = nn.NLLLoss(weight=class_weights)(log_softmax, y)
                    loss = loss_1 + args.alpha * loss_fairness
            else:
                loss = nn.NLLLoss(weight=class_weights)(log_softmax, y)

            epoch_loss += loss.item()

            loss.backward(retain_graph=True)
            optimizer.step()

            _, preds = torch.max(outputs.data, 1)
            epoch_acc += torch.sum(preds == y).item()

        # scheduler.step()

        epoch_loss /= len(finetuneloader)
        epoch_loss_fairness /= len(finetuneloader)
        epoch_acc /= len(finetune_dataset)
        losses.append(epoch_loss)
        print(
            "FINETUNE Epoch %d/%d   Loss_1: %.4f   Loss_2: %.4f   Accuracy: %.4f"
            % (epoch, args.ft_epoch, epoch_loss, epoch_loss_fairness, epoch_acc)
        )

        # Early Stop
        # if (epoch > 50) and (losses[-1] >= losses[-2]):
        #     trigger_times += 1
        #     if trigger_times > 2:
        #         break
        # else:
        #     trigger_times = 0

    #     if epoch_loss < best_loss and epoch > 20:
    #         best_model = deepcopy(model)
    #         best_loss = epoch_loss

    # model = best_model

    model.eval()

    ######
    # Test
    ######
    def get_pred(x):  # Avoid exceeding the memory limit
        dataset = TensorDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        outs = []
        for x in loader:
            out = F.softmax(model.out_fc(x[0].to(device)), dim=1).cpu().detach().numpy()
            outs.append(out)
        outs = np.concatenate(outs)
        pred = np.argmax(outs, 1)
        return outs[:, 1], pred

    out_train, pred_train = get_pred(x_train)
    out_finetune, pred_finetune = get_pred(x_finetune)
    out_test, pred_test = get_pred(x_test)
    y_train, y_finetune, y_test = y_train.numpy(), y_finetune.numpy(), y_test.numpy()
    a_train, a_finetune, a_test = a_train.numpy(), a_finetune.numpy(), a_test.numpy()

    print("== Constrainted ==")
    print("\n-----------------------------------------------------------------------------------\n")

    y_test = torch.from_numpy(y_test)
    print(y_test.shape)
    print(y_test)

    pred_test = torch.from_numpy(pred_test)
    print(pred_test.shape)
    print(pred_test)

    a_test = torch.from_numpy(a_test)
    print(a_test.shape)
    print(a_test)

    ret = get_all_metrics(y_true=y_test, y_pred=pred_test, sensitive_features=a_test)
    print_all_metrics(ret=ret)

    return ret


def main():
    for r in range(repeat_time):
        print(f"Repeated experiment: {r+1}")

        model = MyResNet(num_classes=10, pretrain=False)
        model = model.cuda()
        criterion = nn.NLLLoss()

        start_time = time.time()

        if args.checkpoint is not None:
            ckpt_path = args.checkpoint.replace(".pkl", f"_repeat{r+1}.pkl")
            print("Recovering from %s ..." % (ckpt_path))
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            #########
            # Training #
            #########
            NUM_EPOCHS = 100
            losses = []
            trigger_times = 0
            best_loss = 1e9
            best_model = None
            # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
            optimizer = optim.NAdam(model.parameters(), lr=1e-3, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, last_epoch=-1)
            for epoch in range(NUM_EPOCHS):
                train_per_epoch(model, optimizer, criterion, epoch + 1, NUM_EPOCHS)
                epoch_loss = valid_per_epoch(model, epoch + 1, NUM_EPOCHS, criterion)
                losses.append(epoch_loss)
                if epoch_loss < best_loss and epoch > 5:
                    best_model = deepcopy(model)
                    best_loss = epoch_loss
                # Early Stop
                if (epoch > 20) and (losses[-1] >= losses[-2]):
                    trigger_times += 1
                    if trigger_times > 3:
                        break
                else:
                    trigger_times = 0

                scheduler.step()

            checkpoint = {"model_state_dict": best_model.state_dict()}
            ckpt_path = Path("./checkpoints/cifar10s")
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                checkpoint,
                f"{str(ckpt_path)}/ckpt_cifar10s_{args.constraint.lower()}_seed{str(args.seed)}_repeat{str(r+1)}.pkl",
            )
            model = best_model

        ################
        # Finetune and test #
        ################
        ret = Finetune(model, criterion, train_loader, val_loader, test_loader)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Total training time: {total_time_str}")

        ret["time per epoch"] = total_time / args.ft_epoch
        for i in range(len(metric_index)):
            results[metric_index[i]].append(ret[metric_index[i]])

    for m_index in metric_index:
        fout.write(m_index + "\t")
        for i in range(repeat_time):
            fout.write("%f\t" % results[m_index][i])
        fout.write("%f\t%f\n" % (mean(results[m_index]), std(results[m_index])))
    fout.close()


if __name__ == "__main__":
    main()
