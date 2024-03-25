import os
import sys
import time
import torch
import warnings
from pathlib import Path

from torch import nn
from numpy import mean, std
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from torch.cuda.amp import autocast, GradScaler

project_dir = "/root/DL-Fairness-Study"
warnings.filterwarnings("ignore")
sys.path.insert(1, os.path.join(project_dir, "FR"))

from utils import *
from models.model_zoo import *
from models.resnet9 import resnet9
from datasets.celeba import MyCelebA
from datasets.utkface import UTKFace
from datasets.cifar10s import CIFAR_10S

sys.path.insert(1, project_dir)

from arguments import get_args
from metrics import get_metric_index, get_all_metrics, print_all_metrics


def fairness_evaluation(reprogram, test_loader, predictor, epoch):
    if reprogram is not None:
        reprogram.eval()
    predictor.eval()
    pbar = tqdm(test_loader, total=len(test_loader), ncols=120, desc="Fairness Evaluation")
    fxs = []
    y_all = []
    d_all = []
    test_true_num = 0
    test_total_num = 0
    for x, y, d in pbar:
        y_all.append(y)
        d_all.append(d)
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            if reprogram is not None:
                x = reprogram(x)

            lgt = predictor(x)
        test_total_num += y.shape[0]
        pred = lgt.argmax(1)  # [bs]
        fxs.append(pred)
        test_true_num += (pred == y.view(-1)).type(torch.float).sum().detach().cpu().item()
        acc = test_true_num * 1.0 / test_total_num
        pbar.set_description(f"Test Epoch {epoch} Acc {100 * acc:.6f}%")

    pbar.set_description(f"Test Epoch {epoch} Acc {100 * test_true_num / test_total_num:.6f}%")
    y_all, d_all = torch.cat(y_all).view(-1).cpu(), torch.cat(d_all).view(-1).cpu()
    assert y_all.shape[0] == test_total_num

    fxs = torch.cat(fxs).view(-1).detach().cpu()
    ret = get_all_metrics(y_true=y_all, y_pred=fxs, sensitive_features=d_all)
    print_all_metrics(ret=ret)

    return ret, test_true_num / test_total_num


def main(args):
    setup_seed(args.seed)

    # Sanity Check!
    assert args.rmethod in ["std", "adv"] or args.model_path is not None

    # metric_index = get_metric_index()
    # metric_index.remove("time per epoch")
    # print(metric_index)

    # if args.checkpoint and args.rmethod in ["repro", "rpatch"]:
    #     result_dir = f"/root/study/results/{args.dataset}"
    #     result_path = Path(result_dir)
    #     result_path.mkdir(parents=True, exist_ok=True)

    #     if args.dataset == "celeba":
    #         if "Male" in args.domain_attrs and "Blond_Hair" in args.target_attrs:
    #             if args.adversary_with_y:
    #                 fout = open("/".join([str(result_path), f"fair_reprogram_{args.result_dir}_eo_gender_blond_hair.txt"]), "w")
    #             else:
    #                 fout = open("/".join([str(result_path), f"fair_reprogram_{args.result_dir}_dp_gender_blond_hair.txt"]), "w")
    #     elif args.dataset == "utkface":
    #         if "Age" in args.domain_attrs:
    #             if args.adversary_with_y:
    #                 fout = open("/".join([str(result_path), f"fair_reprogram_{args.result_dir}_eo_age_gender.txt"]), "w")
    #             else:
    #                 fout = open("/".join([str(result_path), f"fair_reprogram_{args.result_dir}_dp_age_gender.txt"]), "w")
    #         elif "Race" in args.domain_attrs:
    #             if args.adversary_with_y:
    #                 fout = open("/".join([str(result_path), f"fair_reprogram_{args.result_dir}_eo_race_gender.txt"]), "w")
    #             else:
    #                 fout = open("/".join([str(result_path), f"fair_reprogram_{args.result_dir}_dp_race_gender.txt"]), "w")
    #     elif args.dataset == "cifar10s":
    #         if args.adversary_with_y:
    #             fout = open("/".join([str(result_path), f"fair_reprogram_{args.result_dir}_eo.txt"]), "w")
    #         else:
    #             fout = open("/".join([str(result_path), f"fair_reprogram_{args.result_dir}_dp.txt"]), "w")
    #     else:
    #         raise AttributeError

    #     results = {}
    #     for m_index in metric_index:
    #         results[m_index] = []

    # make save path dir
    rmethod = {"std": "std", "repro": "border", "rpatch": "patch"}
    if args.dataset in ["celeba", "cifar10s"]:
        model_attr_name = f"fr_{rmethod[args.rmethod]}-{args.dataset}-lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-seed{args.seed}"
    elif args.dataset == "utkface":
        model_attr_name = (
            f"fr_{rmethod[args.rmethod]}-{args.dataset}_{args.sensitive}-lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-seed{args.seed}"
        )

    if args.rmethod in ["adv", "repro", "rpatch", "rpoptim"]:
        model_attr_name += f"-lambda{args.lmbda}"
        model_attr_name += "-eo" if args.adversary_with_y else "-dp"
        if args.rmethod in ["repro", "rpatch", "rpoptim"]:
            model_attr_name += f"-size{args.reprogram_size}"

    if args.trigger_data_num > 0:
        model_attr_name += f"-num{args.trigger_data_num}"

    output_dir = f"{project_dir}/checkpoints/{model_attr_name}"
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    image_size = 32 if args.dataset == "cifar10s" else 224
    transform_train, transform_test = get_transform(
        dataset=args.dataset, method=args.rmethod, image_size=image_size, reprogram_size=args.reprogram_size
    )

    # We will use this a lot.
    use_reprogram = args.rmethod in ["repro", "rpatch", "roptim"]
    use_adv = args.rmethod in ["adv", "repro", "rpatch", "roptim"]

    num_class = 10 if args.dataset == "cifar10s" else 2
    attr_class = 2

    # add dropout
    dropout_prob = 0.5
    classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=dropout_prob), nn.Linear(256, num_class))

    # init model
    if args.model == "resnet18":
        predictor = resnet18(pretrained=False)
        predictor.fc = classifier
        # predictor.fc = nn.Linear(512, num_class)
    elif args.model == "resnet9":
        predictor = resnet9(num_classes=num_class)
    else:
        predictor = resnet20s(num_class)
    predictor = predictor.cuda()

    fc_params = list(map(id, predictor.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, predictor.parameters())
    params = [{"params": base_params, "lr": args.lr}, {"params": predictor.fc.parameters(), "lr": args.lr * 10}]

    p_optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    # p_optim = torch.optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=args.wd)
    p_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        p_optim, gamma=0.1, milestones=[int(0.3 * args.epochs), int(0.6 * args.epochs), int(0.9 * args.epochs)]
    )

    # All the methods except for "std" need adversary
    if use_adv:
        adversary = Adversary(
            input_dim=num_class,
            output_dim=attr_class,
            with_y=args.adversary_with_y,
            with_logits=args.adversary_with_logits,
            use_mlp=True,
        )
        adversary = adversary.cuda()
        a_optim = torch.optim.Adam(adversary.parameters(), lr=args.adversary_lr, weight_decay=args.wd)
        a_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            a_optim, gamma=0.1, milestones=[int(0.3 * args.epochs), int(0.6 * args.epochs), int(0.9 * args.epochs)]
        )
    else:
        adversary = None
        a_optim = None
        a_lr_scheduler = None

    # Initialize reprogrammers
    if use_reprogram:
        reprogram = get_reprogram(method=args.rmethod, image_size=image_size, reprogram_size=args.reprogram_size)
        r_optim = torch.optim.Adam(reprogram.parameters(), lr=args.lr, weight_decay=args.wd)
        r_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            r_optim, gamma=0.1, milestones=[int(0.3 * args.epochs), int(0.6 * args.epochs), int(0.9 * args.epochs)]
        )
    else:
        # We did so because we need to input reprogram into the eval function
        # We create reprogram here for std/adv mode to simplify the call of eval function
        reprogram = None
        r_optim = None
        r_lr_scheduler = None

    # Load checkpoints
    start_epoch = 0
    best_metric = 0.0 if args.rmethod == "std" else 1.0
    if args.model_path is not None:
        if not args.checkpoint:
            checkpoint = torch.load(f"{project_dir}/checkpoints/{args.model_path}/best_model.pth.tar")
        else:
            checkpoint = torch.load(f"{project_dir}/checkpoints/{args.model_path}/best_model.pt")
        predictor.load_state_dict(checkpoint["predictor"])
        # We use the args.resume to distinguish whether the user want to resume the checkpoint
        # or they just want to load the pretrained models and train the reprogram from scratch.
        if args.resume:
            p_optim.load_state_dict(checkpoint["p_optim"])
            p_lr_scheduler.load_state_dict(checkpoint["p_lr_scheduler"])
            best_metric = checkpoint["best_metric"]
            start_epoch = checkpoint["epoch"]
            if use_adv:
                adversary.load_state_dict(checkpoint["adversary"])
                a_optim.load_state_dict(checkpoint["a_optim"])
                a_lr_scheduler.load_state_dict(checkpoint["a_lr_scheduler"])
            if use_reprogram:
                reprogram.load_state_dict(checkpoint["reprogram"])
                r_optim.load_state_dict(checkpoint["r_optim"])
                r_lr_scheduler.load_state_dict(checkpoint["r_lr_scheduler"])

    data_dir = f"{project_dir}/data"
    dataset_dir = f"{data_dir}/{args.dataset}"
    if args.dataset == "celeba":
        train_set = MyCelebA(root=data_dir, split="train", transform=transform_train, target_attr=args.target)
        val_set = MyCelebA(root=data_dir, split="valid", transform=transform_test, target_attr=args.target)
        test_set = MyCelebA(root=data_dir, split="test", transform=transform_test, target_attr=args.target)
    elif args.dataset == "utkface":
        bias_attr = args.sensitive
        print(bias_attr)
        train_set = UTKFace(root=dataset_dir, split="train", transform=transform_train, bias_attr=bias_attr)
        val_set = UTKFace(root=dataset_dir, split="valid", transform=transform_test, bias_attr=bias_attr)
        test_set = UTKFace(root=dataset_dir, split="test", transform=transform_test, bias_attr=bias_attr)
    elif args.dataset == "cifar10s":
        train_set = CIFAR_10S(root=dataset_dir, split="train", transform=transform_train, skew_ratio=0.95)
        val_set = CIFAR_10S(root=dataset_dir, split="valid", transform=transform_test, skew_ratio=0.95)
        test_set = CIFAR_10S(root=dataset_dir, split="test", transform=transform_test, skew_ratio=0.95)
    else:
        raise ValueError

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.checkpoint or args.model_path is not None:
        print("================= Evaluating on Test Set before Training =================")
        test_result, test_acc = fairness_evaluation(None, test_loader, predictor, -1)
        print(f"Test Accuracy: {test_acc}")
        print(test_result)
        if args.checkpoint:
            sys.exit()

    time_per_epoch = []
    scaler = GradScaler()

    for epoch in range(start_epoch, args.epochs):
        # training
        if args.rmethod == "std":
            predictor.train()
        elif args.rmethod == "adv":
            predictor.train()
            adversary.train()
        else:
            # Freeze the predictor for sure.
            predictor.eval()
            for param in predictor.parameters():
                param.requires_grad = False
            adversary.train()
            reprogram.train()
        start = time.time()
        print(f"======================================= Epoch {epoch} =======================================")
        pbar = tqdm(train_loader, total=len(train_loader), ncols=120)
        total_num = 0
        true_num = 0
        for x, y, d in pbar:
            x, y, d = x.cuda(), y.cuda(), d.cuda()
            y_one_hot = get_one_hot(y, num_class)  # one-hot [bs, num_class]
            d_one_hot = get_one_hot(d, attr_class)  # one-hot [bs, attr_class]

            p_optim.zero_grad()
            if args.rmethod != "std":
                a_optim.zero_grad()
                if args.rmethod != "adv":
                    r_optim.zero_grad()

            with autocast():
                if reprogram is not None:
                    x = reprogram(x)

                lgt = predictor(x)
                # print(lgt.shape)
                pred_loss = nn.functional.cross_entropy(lgt, y_one_hot)
                if args.rmethod != "std":
                    protect_pred = adversary(lgt, y=y_one_hot if args.adversary_with_y else None)
                    adversary_loss = torch.nn.functional.cross_entropy(protect_pred, d_one_hot)

            if use_adv:
                # The target of the adversary
                working_model = reprogram if use_reprogram else predictor

                scaler.scale(adversary_loss).backward(retain_graph=True)
                adversary_grad = {name: param.grad.clone() for name, param in working_model.named_parameters()}
                scaler.step(a_optim)
                scaler.update()
                if not use_reprogram:
                    p_optim.zero_grad()
                scaler.scale(pred_loss).backward()
                with torch.no_grad():
                    for name, param in working_model.named_parameters():
                        if name in adversary_grad.keys():
                            unit_adversary_grad = adversary_grad[name] / torch.linalg.norm(adversary_grad[name])
                            param.grad -= (param.grad * unit_adversary_grad).sum() * unit_adversary_grad
                            param.grad -= args.lmbda * adversary_grad[name]
                    del adversary_grad
                if use_reprogram:
                    scaler.step(r_optim)
                    scaler.update()
                else:
                    scaler.step(p_optim)
                    scaler.update()
            else:
                scaler.scale(pred_loss).backward()
                scaler.step(p_optim)
                scaler.update()

            # results for this batch
            total_num += y.size(0)
            true_num += (lgt.argmax(1) == y.view(-1)).type(torch.float).sum().detach().cpu().item()
            acc = true_num * 1.0 / total_num
            pbar.set_description(f"Training Epoch {epoch} Acc {100 * acc:.2f}%")
        pbar.set_description(f"Training Epoch {epoch} Acc {100 * true_num / total_num:.2f}%")

        if args.rmethod == "std":
            p_lr_scheduler.step()
        elif args.rmethod == "adv":
            p_lr_scheduler.step()
            a_lr_scheduler.step()
        else:
            a_lr_scheduler.step()
            r_lr_scheduler.step()

        # evaluating
        print("================= Evaluating on Validation Set =================")
        res, accuracy = fairness_evaluation(reprogram, val_loader, predictor, epoch)
        print(res)
        print(accuracy)

        if args.rmethod == "std":
            metric = res["bacc"]
            if metric > best_metric:
                print("+++++++++++ Find New Best Metric +++++++++++")
                best_metric = metric
                cp = {
                    "predictor": predictor.state_dict(),
                    "p_optim": p_optim.state_dict(),
                    "p_lr_scheduler": p_lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                }
                if args.rmethod != "std":
                    cp["adversary"] = adversary.state_dict()
                    cp["a_optim"] = a_optim.state_dict()
                    cp["a_lr_scheduler"] = a_lr_scheduler.state_dict()
                    if args.rmethod != "adv":
                        cp["reprogram"] = reprogram.state_dict()
                        cp["r_optim"] = r_optim.state_dict()
                        cp["r_lr_scheduler"] = r_lr_scheduler.state_dict()
                torch.save(cp, f"{save_path}/best_model.pt")
        else:
            metric = res["aaod"]
            if metric < best_metric:
                print("+++++++++++ Find New Best Metric +++++++++++")
                best_metric = metric
                cp = {
                    "predictor": predictor.state_dict(),
                    "p_optim": p_optim.state_dict(),
                    "p_lr_scheduler": p_lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                }
                if args.rmethod != "std":
                    cp["adversary"] = adversary.state_dict()
                    cp["a_optim"] = a_optim.state_dict()
                    cp["a_lr_scheduler"] = a_lr_scheduler.state_dict()
                    if args.rmethod != "adv":
                        cp["reprogram"] = reprogram.state_dict()
                        cp["r_optim"] = r_optim.state_dict()
                        cp["r_lr_scheduler"] = r_lr_scheduler.state_dict()
                torch.save(cp, f"{save_path}/best_model.pt")
        print("================= Evaluating on Test Set =================")
        test_result_list, accuracy = fairness_evaluation(reprogram, test_loader, predictor, epoch)
        print(test_result_list)
        print(accuracy)

        end = time.time()
        print(f"Time Consumption for one epoch is {end - start}s")
        time_per_epoch.append(end - start)

    print(f"Average Time Consumption for one epoch is {mean(time_per_epoch)}s")


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
