import os
import sys
import math
import argparse
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(1, "/root/study")

from helper import get_metric_list


def get_data(dataset, sensitive, target, methods):
    acc = []
    bacc = []
    for m in methods:
        if dataset == "cifar10s":
            filename = os.path.join(f"/root/study/results/{dataset}", f"{m}.txt")
        else:
            filename = os.path.join(f"/root/study/results/{dataset}", f"{m}_{sensitive}_{target}.txt")

        acc_list = get_metric_list(filename, metric="acc")
        bacc_list = get_metric_list(filename, metric="bacc")
        acc.append(acc_list)
        bacc.append(bacc_list)

    # print(acc)
    # print(bacc)

    return acc, bacc


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["celeba", "utkface", "cifar10s"], help="Dataset name")
    parser.add_argument("-s", "--sensitive", default="age", type=str, help="Sensitive attribute")
    parser.add_argument("-t", "--target", default="blond_hair", type=str, help="Target attribute")
    opt = parser.parse_args()

    return opt


def main(opt):
    methods = ["us", "os", "uw", "bm_none", "adv", "di", "bc_bb", "flac", "mfd", "fdr_eo", "fr_border", "fr_patch", "faap"]

    acc, bacc = get_data(opt.dataset, opt.sensitive, opt.target, methods)
    labels = ["US", "OS", "UW", "BM", "Adv", "DI", "BC+BB", "FLAC", "MFD", "FDR", "FR-B", "FR-P", "FAAP"]
    # colors = [
    #     (0 / 255.0, 191 / 255.0, 255 / 255.0),  # 深蓝
    #     (255 / 255.0, 165 / 255.0, 0 / 255.0),  # 橙色
    #     (255 / 255.0, 255 / 255.0, 0 / 255.0),  # 黄色
    #     (0 / 255.0, 128 / 255.0, 0 / 255.0),  # 绿色
    #     (0 / 255.0, 0 / 255.0, 255 / 255.0),  # 蓝色
    #     (255 / 255.0, 0 / 255.0, 255 / 255.0),  # 粉色
    #     (0 / 255.0, 255 / 255.0, 255 / 255.0),  # 青色
    #     (128 / 255.0, 0 / 255.0, 128 / 255.0),  # 紫色
    #     (139 / 255.0, 69 / 255.0, 19 / 255.0),  # 棕色
    #     (128 / 255.0, 128 / 255.0, 128 / 255.0),  # 灰色
    #     (0 / 255.0, 139 / 255.0, 139 / 255.0),  # 青绿色
    #     (139 / 255.0, 0 / 255.0, 0 / 255.0),  # 深红色
    #     (128 / 255.0, 0 / 255.0, 0 / 255.0),  # 栗色
    # ]

    plt.figure(figsize=(20, 20))
    plt.rc("font", family="Times New Roman", size=54)

    # 选择颜色映射
    cmap = plt.get_cmap("tab20")  # 可以替换为其他颜色映射

    selected_colors = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 16, 18]

    # 获取一组颜色
    colors = [cmap(i) for i in selected_colors]  # 选择数量符合你需要的颜色数量

    pos_acc = [1 + i * 0.4 for i in range(len(methods))]
    boxplot_acc = plt.boxplot(acc, positions=pos_acc, widths=0.3, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_acc["boxes"], colors):
        patch.set_facecolor(color)
    # # 使用默认颜色循环为每个箱线图分配颜色
    # for box in boxplot_acc["boxes"]:
    #     box.set_facecolor(next(plt.gca()._get_lines.prop_cycler)["color"])

    start_bacc = 1.5 + float(math.ceil(pos_acc[-1]))
    pos_bacc = [start_bacc + i * 0.4 for i in range(len(methods))]
    boxplot_bacc = plt.boxplot(bacc, positions=pos_bacc, widths=0.3, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot_bacc["boxes"], colors):
        patch.set_facecolor(color)
    # # 使用默认颜色循环为每个箱线图分配颜色
    # for box in boxplot_bacc["boxes"]:
    #     box.set_facecolor(next(plt.gca()._get_lines.prop_cycler)["color"])

    x_position = [1, start_bacc]
    x_position_metrics = ["Accuracy", "Balanced Accuracy"]
    plt.xticks([i + 2.4 for i in x_position], x_position_metrics)

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

    # legend = plt.legend(boxplot_acc["boxes"], labels, loc="best")

    if opt.dataset == "utkface":
        plt.savefig(fname=f"boxplot_acc_{opt.dataset}_{opt.sensitive}.pdf", bbox_inches="tight")
    else:
        plt.savefig(fname=f"boxplot_acc_{opt.dataset}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
