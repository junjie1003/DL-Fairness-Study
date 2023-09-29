import os
import sys
import math
import argparse
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(1, "/root/study")

from helper import get_metric_list


def get_data(dataset, sensitive, target, methods):
    method_dict = {}
    for m in methods:
        method_dict[m] = []
        if dataset == "cifar10s":
            filename = os.path.join(f"results/{dataset}", f"{m}.txt")
        else:
            filename = os.path.join(f"results/{dataset}", f"{m}_{sensitive}_{target}.txt")

        acc_list = get_metric_list(filename, metric="acc")
        bacc_list = get_metric_list(filename, metric="bacc")
        method_dict[m].append(acc_list)
        method_dict[m].append(bacc_list)

    return method_dict


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["celeba", "utkface", "cifar10s"], help="Dataset name")
    parser.add_argument("-s", "--sensitive", default="age", type=str, help="Sensitive attribute")
    parser.add_argument("-t", "--target", default="blond_hair", type=str, help="Target attribute")
    opt = parser.parse_args()

    return opt


def main(opt):
    methods = ["us", "os", "uw", "bm_none", "adv", "di", "bc_bb", "flac", "mfd", "fdr_eo", "fr_border", "fr_patch", "faap"]

    method_dict = get_data(opt.dataset, opt.sensitive, opt.target, methods)
    labels = ["Accuracy", "Balanced Accuracy"]

    plt.figure(figsize=(20, 12))
    plt.rc("font", family="Times New Roman", size=48)

    # 选择颜色映射
    cmap = plt.get_cmap("tab20")  # 可以替换为其他颜色映射

    selected_colors = [1, 2]

    # 获取一组颜色
    colors = [cmap(i) for i in selected_colors]  # 选择数量符合你需要的颜色数量

    # US
    pos_us = [0.2 + i * 0.7 for i in range(len(labels))]
    boxplot_us = plt.boxplot(method_dict["us"], positions=pos_us, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_us["boxes"], colors):
        patch.set_facecolor(color)

    start_os = 0.8 + float(math.ceil(pos_us[-1]))
    pos_os = [start_os + i * 0.7 for i in range(len(labels))]
    boxplot_os = plt.boxplot(method_dict["os"], positions=pos_os, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_os["boxes"], colors):
        patch.set_facecolor(color)

    start_uw = 0.8 + float(math.ceil(pos_os[-1]))
    pos_uw = [start_uw + i * 0.7 for i in range(len(labels))]
    boxplot_uw = plt.boxplot(method_dict["uw"], positions=pos_uw, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_uw["boxes"], colors):
        patch.set_facecolor(color)

    start_bm = 0.8 + float(math.ceil(pos_uw[-1]))
    pos_bm = [start_bm + i * 0.7 for i in range(len(labels))]
    boxplot_bm = plt.boxplot(method_dict["bm_none"], positions=pos_bm, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_bm["boxes"], colors):
        patch.set_facecolor(color)

    start_adv = 0.8 + float(math.ceil(pos_bm[-1]))
    pos_adv = [start_adv + i * 0.7 for i in range(len(labels))]
    boxplot_adv = plt.boxplot(method_dict["adv"], positions=pos_adv, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_adv["boxes"], colors):
        patch.set_facecolor(color)

    start_di = 0.8 + float(math.ceil(pos_adv[-1]))
    pos_di = [start_di + i * 0.7 for i in range(len(labels))]
    boxplot_di = plt.boxplot(method_dict["di"], positions=pos_di, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_di["boxes"], colors):
        patch.set_facecolor(color)

    start_bc_bb = 0.8 + float(math.ceil(pos_di[-1]))
    pos_bc_bb = [start_bc_bb + i * 0.7 for i in range(len(labels))]
    boxplot_bc_bb = plt.boxplot(method_dict["bc_bb"], positions=pos_bc_bb, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_bc_bb["boxes"], colors):
        patch.set_facecolor(color)

    start_flac = 0.8 + float(math.ceil(pos_bc_bb[-1]))
    pos_flac = [start_flac + i * 0.7 for i in range(len(labels))]
    boxplot_flac = plt.boxplot(method_dict["flac"], positions=pos_flac, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_flac["boxes"], colors):
        patch.set_facecolor(color)

    start_mfd = 0.8 + float(math.ceil(pos_flac[-1]))
    pos_mfd = [start_mfd + i * 0.7 for i in range(len(labels))]
    boxplot_mfd = plt.boxplot(method_dict["mfd"], positions=pos_mfd, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_mfd["boxes"], colors):
        patch.set_facecolor(color)

    start_fdr = 0.8 + float(math.ceil(pos_mfd[-1]))
    pos_fdr = [start_fdr + i * 0.7 for i in range(len(labels))]
    boxplot_fdr = plt.boxplot(method_dict["fdr_eo"], positions=pos_fdr, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_fdr["boxes"], colors):
        patch.set_facecolor(color)

    start_fr_b = 0.8 + float(math.ceil(pos_fdr[-1]))
    pos_fr_b = [start_fr_b + i * 0.7 for i in range(len(labels))]
    boxplot_fr_b = plt.boxplot(method_dict["fr_border"], positions=pos_fr_b, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_fr_b["boxes"], colors):
        patch.set_facecolor(color)

    start_fr_p = 0.8 + float(math.ceil(pos_fr_b[-1]))
    pos_fr_p = [start_fr_p + i * 0.7 for i in range(len(labels))]
    boxplot_fr_p = plt.boxplot(method_dict["fr_patch"], positions=pos_fr_p, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_fr_p["boxes"], colors):
        patch.set_facecolor(color)

    start_faap = 0.8 + float(math.ceil(pos_fr_p[-1]))
    pos_faap = [start_faap + i * 0.7 for i in range(len(labels))]
    boxplot_faap = plt.boxplot(method_dict["faap"], positions=pos_faap, widths=0.6, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_faap["boxes"], colors):
        patch.set_facecolor(color)

    x_position = [
        0.3,
        start_os,
        start_uw,
        start_bm,
        start_adv,
        start_di,
        start_bc_bb,
        start_flac,
        start_mfd,
        start_fdr,
        start_fr_b,
        start_fr_p,
        start_faap,
    ]
    x_position_metrics = ["US", "OS", "UW", "BM", "Adv", "DI", "BC+BB", "FLAC", "MFD", "FDR", "FR-B", "FR-P", "FAAP"]
    plt.xticks([i + 0.3 for i in x_position], x_position_metrics, rotation=45)

    if opt.dataset == "celeba":
        y_major_locator = plt.MultipleLocator(0.04)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        # plt.ylim(0.72, 0.96)
    elif opt.dataset == "utkface":
        if opt.sensitive == "age":
            y_major_locator = plt.MultipleLocator(0.04)
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            plt.ylim(0.73, 0.93)
        elif opt.sensitive == "race":
            y_major_locator = plt.MultipleLocator(0.04)
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            plt.ylim(0.73, 0.93)
    elif opt.dataset == "cifar10s":
        y_major_locator = plt.MultipleLocator(0.04)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        # plt.ylim(0.72, 0.96)

    # legend = plt.legend(boxplot_faap["boxes"], labels, loc="best")

    if opt.dataset == "utkface":
        plt.savefig(fname=f"boxplot_acc_{opt.dataset}_{opt.sensitive}.pdf", bbox_inches="tight")
    else:
        plt.savefig(fname=f"boxplot_acc_{opt.dataset}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
