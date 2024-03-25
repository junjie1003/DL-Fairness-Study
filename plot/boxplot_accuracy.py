import os
import sys
import math
import argparse
import matplotlib
import matplotlib.pyplot as plt

project_dir = "/root/DL-Fairness-Study"
sys.path.insert(1, project_dir)
from helper import get_metric_list


plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42

datasets = ["celeba", "utkface", "cifar10s"]
methods = ["us", "os", "uw", "bm", "adv", "di", "bc", "flac", "mfd", "fdr", "fr_border", "fr_patch", "faap"]


for dataset in datasets:
    if dataset == "utkface":
        for sensitive in ["age", "race"]:
            method_dict = {}
            for method in methods:
                method_dict[method] = []
                filename = f"{project_dir}/results/{method}_{dataset}_{sensitive}.txt"
                acc_list = get_metric_list(filename, metric="acc")
                bacc_list = get_metric_list(filename, metric="bacc")
                method_dict[method].append(acc_list)
                method_dict[method].append(bacc_list)

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
            boxplot_bm = plt.boxplot(method_dict["bm"], positions=pos_bm, widths=0.6, patch_artist=True, labels=labels)
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

            start_bc = 0.8 + float(math.ceil(pos_di[-1]))
            pos_bc = [start_bc + i * 0.7 for i in range(len(labels))]
            boxplot_bc = plt.boxplot(method_dict["bc"], positions=pos_bc, widths=0.6, patch_artist=True, labels=labels)
            for patch, color in zip(boxplot_bc["boxes"], colors):
                patch.set_facecolor(color)

            start_flac = 0.8 + float(math.ceil(pos_bc[-1]))
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
            boxplot_fdr = plt.boxplot(method_dict["fdr"], positions=pos_fdr, widths=0.6, patch_artist=True, labels=labels)
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
                start_bc,
                start_flac,
                start_mfd,
                start_fdr,
                start_fr_b,
                start_fr_p,
                start_faap,
            ]
            x_position_metrics = ["US", "OS", "UW", "BM", "Adv", "DI", "BC+BB", "FLAC", "MFD", "FDR", "FR-B", "FR-P", "FAAP"]
            plt.xticks([i + 0.3 for i in x_position], x_position_metrics, rotation=45)

            if sensitive == "age":
                y_major_locator = plt.MultipleLocator(0.04)
                ax = plt.gca()
                ax.yaxis.set_major_locator(y_major_locator)
                plt.ylim(0.73, 0.93)
            elif sensitive == "race":
                y_major_locator = plt.MultipleLocator(0.04)
                ax = plt.gca()
                ax.yaxis.set_major_locator(y_major_locator)
                plt.ylim(0.73, 0.93)

            legend = plt.legend(boxplot_faap["boxes"], labels, loc="best")

            plt.savefig(fname=f"boxplot_acc_{dataset}_{sensitive}.pdf", bbox_inches="tight", format="pdf", dpi=600)

    else:
        method_dict = {}
        for method in methods:
            method_dict[method] = []
            filename = f"{project_dir}/results/{method}_{dataset}.txt"
            acc_list = get_metric_list(filename, metric="acc")
            bacc_list = get_metric_list(filename, metric="bacc")
            method_dict[method].append(acc_list)
            method_dict[method].append(bacc_list)

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
        boxplot_bm = plt.boxplot(method_dict["bm"], positions=pos_bm, widths=0.6, patch_artist=True, labels=labels)
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

        start_bc = 0.8 + float(math.ceil(pos_di[-1]))
        pos_bc = [start_bc + i * 0.7 for i in range(len(labels))]
        boxplot_bc = plt.boxplot(method_dict["bc"], positions=pos_bc, widths=0.6, patch_artist=True, labels=labels)
        for patch, color in zip(boxplot_bc["boxes"], colors):
            patch.set_facecolor(color)

        start_flac = 0.8 + float(math.ceil(pos_bc[-1]))
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
        boxplot_fdr = plt.boxplot(method_dict["fdr"], positions=pos_fdr, widths=0.6, patch_artist=True, labels=labels)
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
            start_bc,
            start_flac,
            start_mfd,
            start_fdr,
            start_fr_b,
            start_fr_p,
            start_faap,
        ]
        x_position_metrics = ["US", "OS", "UW", "BM", "Adv", "DI", "BC+BB", "FLAC", "MFD", "FDR", "FR-B", "FR-P", "FAAP"]
        plt.xticks([i + 0.3 for i in x_position], x_position_metrics, rotation=45)

        if dataset == "celeba":
            y_major_locator = plt.MultipleLocator(0.04)
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            # plt.ylim(0.72, 0.96)
        elif dataset == "cifar10s":
            y_major_locator = plt.MultipleLocator(0.04)
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            # plt.ylim(0.72, 0.96)

        legend = plt.legend(boxplot_faap["boxes"], labels, loc="best")

        plt.savefig(fname=f"boxplot_acc_{dataset}.pdf", bbox_inches="tight", format="pdf", dpi=600)
