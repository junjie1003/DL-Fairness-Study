import os
import torch
import numpy as np


def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(filter(lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix), os.listdir(root)))
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def get_accuracy(outputs, labels, binary=False):
    # if multi-label classification
    if len(labels.size()) > 1:
        outputs = (outputs > 0.0).float()
        correct = ((outputs == labels)).float().sum()
        total = torch.tensor(labels.shape[0] * labels.shape[1], dtype=torch.float)
        avg = correct / total
        return avg.item()
    if binary:
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
    else:
        predictions = torch.argmax(outputs, 1)
    c = (predictions == labels).float().squeeze()
    accuracy = torch.mean(c)
    return accuracy.item()


def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")


def make_log_name(args):
    log_name = f"{args.method}"

    if args.dataset == "utkface":
        log_name += f"-{args.dataset}_{args.sensitive}"
    else:
        log_name += f"-{args.dataset}"

    log_name += f"-lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-seed{args.seed}"

    if args.labelwise:
        log_name += "-labelwise"

    if args.teacher_path is not None:
        log_name += f"-temp{args.kd_temp}-lambh{args.lambh}-lambf{args.lambf}"

        if args.no_annealing:
            log_name += "-fixedlamb"

    if args.pretrained:
        log_name += "-pretrained"

    if args.method == "kd_mfd":
        log_name += "-{}".format(args.kernel)
        log_name += "-sigma{}".format(args.sigma) if args.kernel == "rbf" else ""
        log_name += "-jointfeature" if args.jointfeature else ""

    return log_name
