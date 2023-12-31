import os
import time
import logging
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from numpy import mean, std

import torch
from torch import nn, optim

import sys

sys.path.insert(1, "./")

from flac import flac_loss
from models.resnet import ResNet18
from utils.logging import set_logging
from datasets.cifar10s import get_cifar10s
from utils.utils import AverageMeter, MultiDimAverageMeter, accuracy, load_model, save_model, set_seed, pretty_dict

sys.path.insert(1, "/root/study")
from metrics import get_metric_index, get_all_metrics, print_all_metrics


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--color_classifier",
        type=str,
        default="./bias_capturing_classifiers/bcc_cifar10s.pth",
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bs", type=int, default=128, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1000)
    parser.add_argument("--skewed_ratio", type=float, default=0.95)

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    return opt


def train_per_epoch(model, train_loader, criterion, optimizer):
    model.train()
    avg_loss = AverageMeter()

    train_iter = iter(train_loader)
    for images, labels, biases, _ in tqdm(train_iter, ascii=True):
        bsz = labels.shape[0]
        labels, biases = labels.cuda(), biases.cuda()

        images = images.cuda()
        logits, _ = model(images)

        # Training directly on protected attribute targets
        loss = criterion(logits, biases)

        avg_loss.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss.avg


def validate(model, val_loader, num_classes, bcc=False):
    model.eval()

    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(num_classes, 2))

    with torch.no_grad():
        for images, labels, biases, _ in val_loader:
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()
            bsz = labels.shape[0]

            output, _ = model(images)
            preds = output.data.max(1, keepdim=True)[1].squeeze(1)

            if bcc:
                (acc1,) = accuracy(output, biases, topk=(1,))
                top1.update(acc1[0], bsz)

            else:
                (acc1,) = accuracy(output, labels, topk=(1,))
                top1.update(acc1[0], bsz)

                corrects = (preds == labels).long()
                attrwise_acc_meter.add(corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1))

    return top1.avg, attrwise_acc_meter.get_mean()


def train_vanilla(opt, train_loader, val_loader, test_loader):
    model = ResNet18(pretrained=True).cuda()
    criterion = nn.CrossEntropyLoss()

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]
    print(f"decay_epochs: {decay_epochs}")

    optimizer = torch.optim.NAdam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

    best_accs = 0
    best_epochs = 0
    best_stats = {}
    for epoch in range(1, opt.epochs + 1):
        print(f"Epoch {epoch}")
        print(f"[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}")
        loss = train_per_epoch(model, train_loader, criterion, optimizer)
        print(f"[{epoch} / {opt.epochs}] Loss: {loss}")

        scheduler.step()
        stats = pretty_dict(epoch=epoch)

        valid_accs, _ = validate(model, val_loader, num_classes=2, bcc=True)
        stats["valid/acc"] = valid_accs.item()

        test_accs, _ = validate(model, test_loader, num_classes=2, bcc=True)
        stats["test/acc"] = test_accs.item()

        print(f"[{epoch} / {opt.epochs}] {stats}")

        if stats["test/acc"] > best_accs:
            best_accs = stats["test/acc"]
            best_epochs = epoch
            best_stats = pretty_dict(**{f"best_{k}": v for k, v in stats.items()})

            save_file = opt.color_classifier
            save_model(model, optimizer, opt, epoch, save_file)

        print(f"[{epoch} / {opt.epochs}] best test accuracy: {best_accs:.6f} at epoch {best_epochs} \n best_stats: {best_stats}")


def set_model(opt):
    model = ResNet18(num_classes=10, pretrained=True).cuda()
    criterion = nn.CrossEntropyLoss()

    protected_net = ResNet18()
    protected_net.load_state_dict(load_model(opt.color_classifier))
    protected_net.cuda()

    return model, criterion, protected_net


def train(opt, model, protected_net, train_loader, criterion, optimizer):
    model.train()
    protected_net.eval()

    avg_loss = AverageMeter()
    avg_clloss = AverageMeter()
    avg_miloss = AverageMeter()

    total = 0
    total_b_pred = 0
    train_iter = iter(train_loader)

    for idx, (images, labels, biases, _) in enumerate(tqdm(train_iter)):
        bsz = labels.shape[0]
        labels, biases = labels.cuda(), biases.cuda()

        images = images.cuda()
        logits, features = model(images)
        with torch.no_grad():
            pr_l, pr_feat = protected_net(images)
            predicted_race = pr_l.argmax(dim=1, keepdim=True)

        predicted_race = predicted_race.T
        total_b_pred += predicted_race.eq(biases.view_as(predicted_race)).sum().item()
        total += bsz
        loss_mi_div = opt.alpha * (flac_loss(pr_feat, features, labels, 0.5))
        loss_cl = 0.01 * criterion(logits, labels)
        loss = loss_cl + loss_mi_div

        avg_loss.update(loss.item(), bsz)
        avg_clloss.update(loss_cl.item(), bsz)
        avg_miloss.update(loss_mi_div.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss.avg, avg_clloss.avg, avg_miloss.avg


def main():
    opt = parse_option()

    exp_name = f"flac-cifar10s-{opt.exp_name}-lr{opt.lr}-alpha{opt.alpha}-bs{opt.bs}-seed{opt.seed}"
    opt.exp_name = exp_name

    result_dir = "../results/cifar10s"
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    fout = open("/".join([str(result_path), "flac.txt"]), "w")

    results = {}
    metric_index = get_metric_index()
    for m_index in metric_index:
        results[m_index] = []

    repeat_time = 10

    output_dir = f"results/{exp_name}"
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, "INFO", str(save_path))
    logging.info(f"Set seed: {opt.seed}")
    set_seed(opt.seed)
    logging.info(f"save_path: {save_path}")

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    root = "../data/cifar10s"
    train_loader = get_cifar10s(root, split="train", batch_size=opt.bs, aug=False, skewed_ratio=opt.skewed_ratio)
    val_loader = get_cifar10s(root, split="valid", batch_size=opt.bs, aug=False, skewed_ratio=opt.skewed_ratio)
    test_loader = get_cifar10s(root, split="test", batch_size=opt.bs, aug=False, skewed_ratio=opt.skewed_ratio)

    bcc_path = Path(opt.color_classifier)
    if not bcc_path.exists():
        # Train a vanilla bias-capturing classifier on CIFAR-10S dataset with a standard cross entropy loss
        train_vanilla(opt, train_loader, val_loader, test_loader)

    for r in range(repeat_time):
        logging.info(f"Repeated experiment: {r+1}")

        model, criterion, protected_net = set_model(opt)

        decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]
        logging.info(f"decay_epochs: {decay_epochs}")

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

        (save_path / "checkpoints").mkdir(parents=True, exist_ok=True)

        best_accs = 0
        best_epochs = 0
        best_stats = {}
        start_time = time.time()
        for epoch in range(1, opt.epochs + 1):
            logging.info(f"Epoch {epoch}")
            logging.info(f"[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}")
            loss, cllossp, milossp = train(opt, model, protected_net, train_loader, criterion, optimizer)
            logging.info(f"[{epoch} / {opt.epochs}] Loss: {loss} Loss CE: {cllossp} Loss MI: {milossp}")

            scheduler.step()
            stats = pretty_dict(epoch=epoch)

            valid_accs, valid_attrwise_accs = validate(model, val_loader, num_classes=10, bcc=False)
            stats["valid/acc"] = valid_accs.item()
            stats["valid/acc_unbiased"] = torch.mean(valid_attrwise_accs).item() * 100

            test_accs, test_attrwise_accs = validate(model, test_loader, num_classes=10, bcc=False)
            stats["test/acc"] = test_accs.item()
            stats["test/acc_unbiased"] = torch.mean(test_attrwise_accs).item() * 100

            logging.info(f"[{epoch} / {opt.epochs}] {stats}")

            if stats["test/acc_unbiased"] > best_accs:
                best_accs = stats["test/acc_unbiased"]
                best_epochs = epoch
                best_stats = pretty_dict(**{f"best_{k}": v for k, v in stats.items()})

                save_file = save_path / "checkpoints" / f"best_test_repeat{r+1}.pth"
                save_model(model, optimizer, opt, epoch, save_file)

            logging.info(f"[{epoch} / {opt.epochs}] best test accuracy: {best_accs:.6f} at epoch {best_epochs} \n best_stats: {best_stats}")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f"Total training time: {total_time_str}")

        save_file = save_path / "checkpoints" / f"last_repeat{r+1}.pth"
        save_model(model, optimizer, opt, opt.epochs, save_file)

        # Evaluation
        model.eval()

        with torch.no_grad():
            all_labels, all_biases, all_preds = [], [], []
            for images, labels, biases, _ in test_loader:
                images = images.cuda()

                output, _ = model(images)
                preds = output.data.max(1, keepdim=True)[1].squeeze(1).cpu()

                all_labels.append(labels)
                all_biases.append(biases)
                all_preds.append(preds)

            fin_labels = torch.cat(all_labels)
            fin_biases = torch.cat(all_biases)
            fin_preds = torch.cat(all_preds)

            ret = get_all_metrics(y_true=fin_labels, y_pred=fin_preds, sensitive_features=fin_biases)
            print_all_metrics(ret=ret)

        ret["time per epoch"] = total_time / opt.epochs
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
