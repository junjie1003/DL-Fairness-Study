import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

project_dir = "/root/DL-Fairness-Study"
sys.path.insert(1, project_dir)
from helper import get_metric_list

plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42

datasets = ["celeba", "utkface", "cifar10s"]
metrics = ["SPD", "DEO", "EOD", "AAOD", "AED"]
methods = ["us", "os", "uw", "bm", "adv", "di", "bc", "flac", "mfd", "fdr", "fr_border", "fr_patch", "faap"]

SPD, DEO, EOD, AAOD, AED = [], [], [], [], []

for method in methods:
    for dataset in datasets:
        for metric in metrics:
            metric = metric.lower()
            if dataset == "utkface":
                filename_age = f"{project_dir}/results/{method}_{dataset}_age.txt"
                filename_race = f"{project_dir}/results/{method}_{dataset}_race.txt"
                metric_list_age = get_metric_list(filename_age, metric)
                metric_list_race = get_metric_list(filename_race, metric)
                if metric == "spd":
                    SPD.extend(metric_list_age)
                    SPD.extend(metric_list_race)
                elif metric == "deo":
                    DEO.extend(metric_list_age)
                    DEO.extend(metric_list_race)
                elif metric == "eod":
                    EOD.extend(metric_list_age)
                    EOD.extend(metric_list_race)
                elif metric == "aaod":
                    AAOD.extend(metric_list_age)
                    AAOD.extend(metric_list_race)
                elif metric == "aed":
                    AED.extend(metric_list_age)
                    AED.extend(metric_list_race)
            else:
                filename = f"{project_dir}/results/{method}_{dataset}.txt"
                metric_list = get_metric_list(filename, metric)
                if metric == "spd":
                    SPD.extend(metric_list)
                elif metric == "deo":
                    DEO.extend(metric_list)
                elif metric == "eod":
                    EOD.extend(metric_list)
                elif metric == "aaod":
                    AAOD.extend(metric_list)
                elif metric == "aed":
                    AED.extend(metric_list)

data = {"SPD": [], "DEO": [], "EOD": [], "AAOD": [], "AED": []}

data["SPD"].append(float("{:.2f}".format(pearsonr(SPD, SPD)[0])))
data["SPD"].append(float("{:.2f}".format(pearsonr(SPD, DEO)[0])))
data["SPD"].append(float("{:.2f}".format(pearsonr(SPD, EOD)[0])))
data["SPD"].append(float("{:.2f}".format(pearsonr(SPD, AAOD)[0])))
data["SPD"].append(float("{:.2f}".format(pearsonr(SPD, AED)[0])))

data["DEO"].append(float("{:.2f}".format(pearsonr(DEO, SPD)[0])))
data["DEO"].append(float("{:.2f}".format(pearsonr(DEO, DEO)[0])))
data["DEO"].append(float("{:.2f}".format(pearsonr(DEO, EOD)[0])))
data["DEO"].append(float("{:.2f}".format(pearsonr(DEO, AAOD)[0])))
data["DEO"].append(float("{:.2f}".format(pearsonr(DEO, AED)[0])))

data["EOD"].append(float("{:.2f}".format(pearsonr(EOD, SPD)[0])))
data["EOD"].append(float("{:.2f}".format(pearsonr(EOD, DEO)[0])))
data["EOD"].append(float("{:.2f}".format(pearsonr(EOD, EOD)[0])))
data["EOD"].append(float("{:.2f}".format(pearsonr(EOD, AAOD)[0])))
data["EOD"].append(float("{:.2f}".format(pearsonr(EOD, AED)[0])))

data["AAOD"].append(float("{:.2f}".format(pearsonr(AAOD, SPD)[0])))
data["AAOD"].append(float("{:.2f}".format(pearsonr(AAOD, DEO)[0])))
data["AAOD"].append(float("{:.2f}".format(pearsonr(AAOD, EOD)[0])))
data["AAOD"].append(float("{:.2f}".format(pearsonr(AAOD, AAOD)[0])))
data["AAOD"].append(float("{:.2f}".format(pearsonr(AAOD, AED)[0])))

data["AED"].append(float("{:.2f}".format(pearsonr(AED, SPD)[0])))
data["AED"].append(float("{:.2f}".format(pearsonr(AED, DEO)[0])))
data["AED"].append(float("{:.2f}".format(pearsonr(AED, EOD)[0])))
data["AED"].append(float("{:.2f}".format(pearsonr(AED, AAOD)[0])))
data["AED"].append(float("{:.2f}".format(pearsonr(AED, AED)[0])))

df = pd.DataFrame.from_dict(data, orient="index", columns=metrics)
print(df)

plt.figure(figsize=(10, 6))
sns.set(style="white", font="Times New Roman", font_scale=2)
sns.heatmap(
    data=df,
    cmap="Blues",
    annot=df,
    fmt=".2f",
    annot_kws={"fontname": "Times New Roman", "fontsize": 20},
    linewidths=0.5,
    mask=np.triu(np.ones_like(df.values, dtype=bool), 1),
)
plt.xticks(fontname="Times New Roman", fontsize=21)
plt.yticks(fontname="Times New Roman", fontsize=21)
plt.tight_layout()

plt.savefig("correlation_matrix.pdf", format="pdf", dpi=600)
