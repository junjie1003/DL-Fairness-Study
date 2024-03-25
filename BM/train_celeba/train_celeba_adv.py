import os
import sys
import time
import torch
import logging
import argparse
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
from numpy import mean, std
import torch.nn.functional as F

project_dir = "/root/DL-Fairness-Study"
sys.path.insert(1, os.path.join(project_dir, "BM"))

from debias.utils.logging import set_logging
from debias.datasets.celeba import get_celeba
from debias.networks.resnet import FCResNet18_Base
from debias.utils.utils import AverageMeter, MultiDimAverageMeter, accuracy, pretty_dict, save_model, set_seed

sys.path.insert(1, project_dir)

from arguments import get_args
from metrics import get_metric_index, get_all_metrics, print_all_metrics


def set_model():
    model = FCResNet18_Base().cuda()
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    class_network = nn.Linear(512, 2).cuda()
    domain_network = nn.Linear(512, 2).cuda()

    return [model, class_network, domain_network], [class_criterion, domain_criterion]


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model[0].train()
    avg_loss = AverageMeter()

    train_iter = iter(train_loader)
    for images, labels, _, biases, _, _ in tqdm(train_iter, ascii=True):
        bsz = labels.shape[0]
        labels, biases = labels.cuda(), biases.cuda()

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        optimizer[2].zero_grad()

        images = images.cuda()
        features = model[0](images)
        class_out = model[1](features)
        domain_out = model[2](features)

        class_loss = criterion[0](class_out, labels)
        domain_loss = criterion[1](domain_out, biases)

        if epoch % opt.training_ratio == 0:
            log_softmax = F.log_softmax(domain_out, dim=1)
            confusion_loss = -log_softmax.mean(dim=1).mean()
            loss = class_loss + opt.alpha * confusion_loss
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()

        else:
            # Update the domain classifier
            domain_loss.backward()
            optimizer[2].step()

        avg_loss.update(class_loss.item(), bsz)

    return avg_loss.avg


def validate(val_loader, model):
    model[0].eval()

    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(2, 2))

    with torch.no_grad():
        for images, labels, _, biases, _, _ in val_loader:
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()
            bsz = labels.shape[0]

            feats = model[0](images)
            output = model[1](feats)

            preds = output.data.max(1, keepdim=True)[1].squeeze(1)

            (acc1,) = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1))

    return top1.avg, attrwise_acc_meter.get_mean(), attrwise_acc_meter.get_acc_diff()


def main():
    opt = get_args()
    opt.target = "blonde" if opt.target == "Blond_Hair" else opt.target
    exp_name = f"adv-celeba-lr{opt.lr}-bs{opt.batch_size}-epochs{opt.epochs}-seed{opt.seed}"

    # result_dir = f'../results/celeba'
    # result_path = Path(result_dir)
    # result_path.mkdir(parents=True, exist_ok=True)
    # if opt.target == 'blonde':
    #     fout = open('/'.join([str(result_path), f'adv_gender_blond_hair.txt']), 'w')

    # results = {}
    # metric_index = get_metric_index()
    # for m_index in metric_index:
    #     results[m_index] = []

    repeat_time = 1

    output_dir = f"{project_dir}/checkpoints/{exp_name}"
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Set seed: {opt.seed}")
    set_seed(opt.seed)
    print(f"save_path: {save_path}")

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    root = f"{project_dir}/data"
    train_loader = get_celeba(root, batch_size=opt.batch_size, target_attr=opt.target, split="train", aug=False)

    val_loaders = {}
    val_loaders["valid"] = get_celeba(root, batch_size=256, target_attr=opt.target, split="train_valid", aug=False)

    val_loaders["test"] = get_celeba(root, batch_size=256, target_attr=opt.target, split="valid", aug=False)

    for r in range(repeat_time):
        # print(f"Repeated experiment: {r+1}")

        model, criterion = set_model()

        # Evaluation
        if opt.checkpoint:
            model[0].load_state_dict(torch.load(f"{save_path}/best_model.pt")["model"])
            model[0].eval()
            model[1].load_state_dict(torch.load(f"{save_path}/best_pred.pt")["model"])
            model[1].eval()

            with torch.no_grad():
                all_labels, all_biases, all_preds = [], [], []
                for images, labels, _, biases, _, _ in val_loaders["test"]:
                    images = images.cuda()

                    feats = model[0](images)
                    output = model[1](feats)

                    preds = output.data.max(1, keepdim=True)[1].squeeze(1).cpu()

                    all_labels.append(labels)
                    all_biases.append(biases)
                    all_preds.append(preds)

                fin_labels = torch.cat(all_labels)
                fin_biases = torch.cat(all_biases)
                fin_preds = torch.cat(all_preds)

                ret = get_all_metrics(y_true=fin_labels, y_pred=fin_preds, sensitive_features=fin_biases)
                print_all_metrics(ret=ret)

            sys.exit()

        #     ret["time per epoch"] = total_time / opt.epochs
        #     for i in range(len(metric_index)):
        #         results[metric_index[i]].append(ret[metric_index[i]])

        # for m_index in metric_index:
        #     fout.write(m_index + "\t")
        #     for i in range(repeat_time):
        #         fout.write("%f\t" % results[m_index][i])
        #     fout.write("%f\t%f\n" % (mean(results[m_index]), std(results[m_index])))
        # fout.close()

        decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]

        optimizer_base = torch.optim.Adam(model[0].parameters(), lr=opt.lr, weight_decay=1e-4)
        optimizer_class = torch.optim.Adam(model[1].parameters(), lr=opt.lr, weight_decay=1e-4)
        optimizer_domain = torch.optim.Adam(model[2].parameters(), lr=opt.lr, weight_decay=1e-4)
        optimizers = [optimizer_base, optimizer_class, optimizer_domain]

        scheduler_base = torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=decay_epochs, gamma=0.1)
        scheduler_class = torch.optim.lr_scheduler.MultiStepLR(optimizers[1], milestones=decay_epochs, gamma=0.1)
        scheduler_domain = torch.optim.lr_scheduler.MultiStepLR(optimizers[2], milestones=decay_epochs, gamma=0.1)
        schedulers = [scheduler_base, scheduler_class, scheduler_domain]

        print(f"decay_epochs: {decay_epochs}")

        best_accs = {"valid": 0, "test": 0}
        best_epochs = {"valid": 0, "test": 0}
        best_stats = {}
        start_time = time.time()
        for epoch in range(1, opt.epochs + 1):
            print(f"[{epoch} / {opt.epochs}] Learning rate: {schedulers[0].get_last_lr()[0]}")
            loss = train(train_loader, model, criterion, optimizers, epoch, opt)
            print(f"[{epoch} / {opt.epochs}] Loss: {loss:.4f}")

            schedulers[0].step()
            schedulers[1].step()
            schedulers[2].step()

            stats = pretty_dict(epoch=epoch)
            for key, val_loader in val_loaders.items():
                accs, valid_attrwise_accs, diff = validate(val_loader, model)

                stats[f"{key}/acc"] = accs.item()
                stats[f"{key}/acc_unbiased"] = torch.mean(valid_attrwise_accs).item() * 100
                stats[f"{key}/diff"] = diff.item() * 100

                eye_tsr = train_loader.dataset.eye_tsr

                stats[f"{key}/acc_skew"] = valid_attrwise_accs[eye_tsr > 0.0].mean().item() * 100
                stats[f"{key}/acc_align"] = valid_attrwise_accs[eye_tsr == 0.0].mean().item() * 100

            print(f"[{epoch} / {opt.epochs}] {valid_attrwise_accs} {stats}")
            for tag in val_loaders.keys():
                if stats[f"{tag}/acc_unbiased"] > best_accs[tag]:
                    best_accs[tag] = stats[f"{tag}/acc_unbiased"]
                    best_epochs[tag] = epoch
                    best_stats[tag] = pretty_dict(**{f"best_{tag}_{k}": v for k, v in stats.items()})

                    # if tag == "valid":
                    #     save_file = save_path / "best_model.pt"
                    #     save_model(model[0], optimizer_base, opt, epoch, save_file)

                print(
                    f"[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats[tag]}"
                )

        save_file_base = save_path / "best_model.pt"
        save_model(model[0], optimizer_base, opt, epoch, save_file_base)
        save_file_class = save_path / "best_pred.pt"
        save_model(model[1], optimizer_class, opt, epoch, save_file_class)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Total training time: {total_time_str}")
        print(f"Time per epoch: {total_time / opt.epochs:.6f}")


if __name__ == "__main__":
    main()
