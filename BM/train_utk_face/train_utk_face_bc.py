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

project_dir = "/root/DL-Fairness-Study"
sys.path.insert(1, os.path.join(project_dir, "BM"))

from debias.utils.logging import set_logging
from debias.networks.resnet import FCResNet18
from debias.datasets.utk_face import get_utk_face
from debias.losses.bias_contrastive import BiasContrastiveLoss
from debias.utils.utils import AverageMeter, MultiDimAverageMeter, accuracy, pretty_dict, save_model, set_seed

sys.path.insert(1, project_dir)

from arguments import get_args
from metrics import get_metric_index, get_all_metrics, print_all_metrics


def set_model(train_loader, opt):
    model = FCResNet18().cuda()
    criterion = BiasContrastiveLoss(confusion_matrix=train_loader.dataset.confusion_matrix, bb=opt.bb)

    return model, criterion


def train(train_loader, cont_train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    avg_ce_loss = AverageMeter()
    avg_con_loss = AverageMeter()
    avg_loss = AverageMeter()

    train_iter = iter(train_loader)
    cont_train_iter = iter(cont_train_loader)
    for images, labels, _, biases, _, _ in tqdm(train_iter, ascii=True):
        try:
            cont_images, cont_labels, _, cont_biases, _, _ = next(cont_train_iter)
        except:
            cont_train_iter = iter(cont_train_loader)
            cont_images, cont_labels, _, cont_biases, _, _ = next(cont_train_iter)

        bsz = labels.shape[0]
        cont_bsz = cont_labels.shape[0]

        labels, biases = labels.cuda(), biases.cuda()

        images = images.cuda()
        logits, _ = model(images)

        total_images = torch.cat([cont_images[0], cont_images[1]], dim=0)
        total_images, cont_labels, cont_biases = total_images.cuda(), cont_labels.cuda(), cont_biases.cuda()
        _, cont_features = model(total_images)

        f1, f2 = torch.split(cont_features, [cont_bsz, cont_bsz], dim=0)
        cont_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        ce_loss, con_loss = criterion(logits, labels, biases, cont_features, cont_labels, cont_biases)

        loss = ce_loss * opt.weight + con_loss

        avg_ce_loss.update(ce_loss.item(), bsz)
        avg_con_loss.update(con_loss.item(), bsz)
        avg_loss.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_ce_loss.avg, avg_con_loss.avg, avg_loss.avg


def validate(val_loader, model):
    model.eval()

    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(2, 2))

    with torch.no_grad():
        for images, labels, _, biases, _, _ in val_loader:
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()
            bsz = labels.shape[0]

            output, _ = model(images)
            preds = output.data.max(1, keepdim=True)[1].squeeze(1)

            (acc1,) = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1))

    return top1.avg, attrwise_acc_meter.get_mean(), attrwise_acc_meter.get_acc_diff()


def main():
    opt = get_args()
    exp_name = f"bc-bb{opt.bb}-utkface_{opt.sensitive}-lr{opt.lr}-bs{opt.batch_size}-cbs{opt.cbs}-epochs{opt.epochs}-w{opt.weight}-ratio{opt.ratio}-aug{opt.aug}-seed{opt.seed}"

    # result_dir = f"../results/utkface"
    # result_path = Path(result_dir)
    # result_path.mkdir(parents=True, exist_ok=True)
    # if opt.sensitive == "age":
    #     fout = open("/".join([str(result_path), "bc_bb_age_gender.txt"]), "w")
    # elif opt.sensitive == "race":
    #     fout = open("/".join([str(result_path), "bc_bb_race_gender.txt"]), "w")

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

    root = f"{project_dir}/data/utkface"
    train_loader = get_utk_face(
        root,
        batch_size=opt.batch_size,
        bias_attr=opt.sensitive,
        split="train",
        aug=False,
    )
    print(
        f"confusion_matrix - \n original: {train_loader.dataset.confusion_matrix_org}, \n normalized: {train_loader.dataset.confusion_matrix}, \n b|y: {train_loader.dataset.confusion_matrix_by}"
    )

    cont_train_loader = get_utk_face(
        root, batch_size=opt.cbs, bias_attr=opt.sensitive, split="train", aug=opt.aug, two_crop=True, ratio=opt.ratio, given_y=True
    )

    val_loaders = {}
    val_loaders["valid"] = get_utk_face(root, batch_size=256, bias_attr=opt.sensitive, split="valid", aug=False)

    val_loaders["test"] = get_utk_face(root, batch_size=256, bias_attr=opt.sensitive, split="test", aug=False)

    for r in range(repeat_time):
        # print(f"Repeated experiment: {r+1}")

        model, criterion = set_model(train_loader, opt)

        # Evaluation
        if opt.checkpoint:
            model.load_state_dict(torch.load(f"{save_path}/best_model.pt")["model"])
            model.eval()

            with torch.no_grad():
                all_labels, all_biases, all_preds = [], [], []
                for images, labels, _, biases, _, _ in val_loaders["test"]:
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

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
        print(f"decay_epochs: {decay_epochs}")

        best_accs = {"valid": 0, "test": 0}
        best_epochs = {"valid": 0, "test": 0}
        best_stats = {}
        start_time = time.time()
        for epoch in range(1, opt.epochs + 1):
            print(f"[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]} weight: {opt.weight}")
            ce_loss, con_loss, loss = train(train_loader, cont_train_loader, model, criterion, optimizer, epoch, opt)
            print(f"[{epoch} / {opt.epochs}] Loss: {loss} CE Loss: {ce_loss} Con Loss: {con_loss}")

            scheduler.step()

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

                print(
                    f"[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats[tag]}"
                )

        save_file = save_path / f"best_model.pt"
        save_model(model, optimizer, opt, epoch, save_file)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Total training time: {total_time_str}")
        print(f"Time per epoch: {total_time / opt.epochs:.6f}")


if __name__ == "__main__":
    main()
