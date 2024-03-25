import os
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn


def set_seed(seed):
    print(f"=======> Using Fixed Random Seed: {seed} <========")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False  # set to False for final report


def get_results(filename, metric):
    print(f"Metric name: {metric}")
    with open(filename, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split("\t")
            if tokens[0] == metric:
                results = [float(i) for i in tokens[1:11]]
                mean_res = np.mean(results)
                std_res = np.std(results)
                break

    return results, mean_res, std_res


def get_metric_list(filename, metric):
    with open(filename, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split("\t")
            if tokens[0] == metric:
                metric_list = [float(i) for i in tokens[1:11]]
                break

    return metric_list


if __name__ == "__main__":
    metric_index = ["acc", "bacc", "spd", "deo", "eod", "aaod", "aed"]
    # metric_index = ["acc", "bacc", "spd", "deo", "eod", "aaod", "aed", "time per epoch"]
    for mi in metric_index:
        # results, mean_res, std_res = get_results(filename="./results/celeba/faap_gender_blond_hair.txt", metric=mi)
        # results, mean_res, std_res = get_results(filename="./results/utkface/faap_age_gender.txt", metric=mi)
        # results, mean_res, std_res = get_results(filename="./results/utkface/faap_race_gender.txt", metric=mi)
        results, mean_res, std_res = get_results(filename="./results/cifar10s/fr_patch.txt", metric=mi)
        print("{:.3f}\t{:.3f}".format(mean_res, std_res))
        print("{:.6f}\t{:.6f}".format(mean_res, std_res))

    # print(get_metric_list(filename="./results/celeba/us_gender_blond_hair.txt", metric="bacc"))
