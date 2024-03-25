import os
import sys
import time
import torch
import networks
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from numpy import mean, std

project_dir = "/root/DL-Fairness-Study"
sys.path.insert(1, os.path.join(project_dir, "MFD"))

import trainer
import data_handler
from utils import check_log_dir, make_log_name

sys.path.insert(1, project_dir)

from helper import set_seed
from arguments import get_args
from metrics import get_metric_index, get_all_metrics, print_all_metrics

args = get_args()


def main():
    args.gpu = int(args.gpu)
    torch.backends.cudnn.enabled = True

    seed = args.seed
    set_seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    log_name = make_log_name(args)
    dataset = args.dataset
    save_dir = os.path.join(project_dir, "checkpoints", log_name)
    if not args.checkpoint:
        check_log_dir(save_dir)

    # if args.checkpoint:
    #     result_dir = f"../results/{args.dataset}"
    #     result_path = Path(result_dir)
    #     result_path.mkdir(parents=True, exist_ok=True)

    #     if args.dataset == "celeba" and args.target == "Blond_Hair":
    #         fout = open("/".join([str(result_path), "mfd_gender_blond_hair.txt"]), "w")

    #     if args.dataset == "utkface":
    #         if args.sensitive == "age":
    #             fout = open("/".join([str(result_path), f"mfd_age_gender.txt"]), "w")
    #         elif args.sensitive == "race":
    #             fout = open("/".join([str(result_path), f"mfd_race_gender.txt"]), "w")

    #     if args.dataset == "cifar10s":
    #         fout = open("/".join([str(result_path), "mfd.txt"]), "w")

    #     results = {}
    #     metric_index = get_metric_index()
    #     metric_index.remove("time per epoch")
    #     for m_index in metric_index:
    #         results[m_index] = []

    ########################## get dataloader ################################

    tmp = data_handler.DataloaderFactory.get_dataloader(
        args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        sensitive=args.sensitive,
        target=args.target,
        skew_ratio=args.skew_ratio,
        labelwise=args.labelwise,
    )
    num_classes, num_groups, train_loader, test_loader = tmp

    # time_per_epoch = []
    for r in range(args.repeat_time):
        # print(f"Repeated experiment: {r+1}")

        ################################## get model ##################################

        model = networks.ModelFactory.get_model(args.model, num_classes, args.img_size, pretrained=args.pretrained)

        if args.parallel:
            model = nn.DataParallel(model)

        model.cuda()

        # if args.model_path is not None:
        # model.load_state_dict(torch.load(args.model_path))

        teacher = None
        if (args.method.startswith("kd") or args.teacher_path is not None) and not args.checkpoint:
            teacher = networks.ModelFactory.get_model(args.model, train_loader.dataset.num_classes, args.img_size)
            if args.parallel:
                teacher = nn.DataParallel(teacher)
            teacher.load_state_dict(torch.load(f"{project_dir}/checkpoints/{args.teacher_path}"))
            teacher.cuda()

        ################################## get trainer ##################################

        if args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif "SGD" in args.optimizer:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        trainer_ = trainer.TrainerFactory.get_trainer(args.method, model=model, args=args, optimizer=optimizer, teacher=teacher)

        ################################## start training or evaluating ##################################
        # log_name_r = log_name + f"_repeat{r+1}" if args.repeat_time > 1 else log_name
        if not args.checkpoint:
            start_t = time.time()
            trainer_.train(train_loader, test_loader, args.epochs)
            end_t = time.time()

            tpe = (end_t - start_t) / args.epochs
            # time_per_epoch.append(tpe)
            print(f"Time per epoch: {tpe:.6f}")

            train_t = int((end_t - start_t) / 60)  # to minutes
            print("Training Time : {} hours {} minutes".format(int(train_t / 60), (train_t % 60)))

            trainer_.save_model(save_dir)

            if args.evalset == "all":
                trainer_.compute_confusion_matrix("train", train_loader.dataset.num_classes, train_loader, save_dir)
                trainer_.compute_confusion_matrix("test", test_loader.dataset.num_classes, test_loader, save_dir)

            elif args.evalset == "train":
                trainer_.compute_confusion_matrix("train", train_loader.dataset.num_classes, train_loader, save_dir)
            else:
                trainer_.compute_confusion_matrix("test", test_loader.dataset.num_classes, test_loader, save_dir)

            print("Done!")

        else:
            print("Evaluation ----------------")
            trainer_.model.load_state_dict(torch.load(os.path.join(project_dir, "checkpoints", args.model_path)))
            print("Trained model loaded successfully")

            trainer_.model.eval()

            with torch.no_grad():
                all_labels, all_groups, all_preds = [], [], []
                for j, eval_data in enumerate(test_loader):
                    inputs, _, groups, classes, _ = eval_data
                    labels = classes
                    inputs = inputs.cuda()

                    outputs = trainer_.model(inputs)
                    preds = torch.argmax(outputs, 1).cpu()

                    all_labels.append(labels)
                    all_groups.append(groups)
                    all_preds.append(preds)

                fin_labels = torch.cat(all_labels)
                fin_groups = torch.cat(all_groups)
                fin_preds = torch.cat(all_preds)

                ret = get_all_metrics(y_true=fin_labels, y_pred=fin_preds, sensitive_features=fin_groups)
                print(ret)
                print_all_metrics(ret=ret)

            # for i in range(len(metric_index)):
            #     results[metric_index[i]].append(ret[metric_index[i]])

    # if args.mode == "train":
    #     print("Average Training Time Per Epoch: {} seconds".format(mean(time_per_epoch)))

    # if args.mode == "eval":
    #     for m_index in metric_index:
    #         fout.write(m_index + "\t")
    #         for i in range(args.repeat_time):
    #             fout.write("%f\t" % results[m_index][i])
    #         fout.write("%f\t%f\n" % (mean(results[m_index]), std(results[m_index])))
    #     fout.close()


if __name__ == "__main__":
    main()
