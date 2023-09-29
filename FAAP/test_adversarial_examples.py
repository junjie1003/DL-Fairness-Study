import os
import sys
import torch
from pathlib import Path
from numpy import mean, std

from arguments import get_args
from models import Generator, ResNet18

sys.path.insert(1, "/root/study")
from datasets.celeba import get_celeba
from datasets.utkface import get_utkface
from datasets.cifar10s import get_cifar10s
from helper import set_seed, make_log_name
from metrics import get_metric_index, get_all_metrics, print_all_metrics

args = get_args()


def test_adv(args):
    print(args)
    set_seed(args.seed)

    save_dir = Path(f"{args.save_dir}/{args.date}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # result_dir = f"../results/{args.dataset}"
    # result_path = Path(result_dir)
    # result_path.mkdir(parents=True, exist_ok=True)

    # if args.dataset == "celeba":
    #     fout = open("/".join([str(result_path), "faap_gender_blond_hair.txt"]), "w")
    # elif args.dataset == "utkface":
    #     fout = open("/".join([str(result_path), f"faap_{args.sensitive}_gender.txt"]), "w")
    # elif args.dataset == "cifar10s":
    #     fout = open("/".join([str(result_path), "faap.txt"]), "w")
    # else:
    #     raise ValueError("Invalid dataset type!")

    # results = {}
    # metric_index = get_metric_index()
    # metric_index.remove("time per epoch")
    # for m_index in metric_index:
    #     results[m_index] = []

    start_epoch = args.epochs - 9

    log_name_d = make_log_name(args, model_name="deployed_model")
    log_name_g = make_log_name(args, model_name="generator")
    num_classes = 10 if args.dataset == "cifar10s" else 2

    # Load datasets
    data_dir = f"{args.data_dir}/{args.dataset}"
    if args.dataset == "celeba":
        train_dataset, train_loader = get_celeba(
            root=args.data_dir, split="train", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
        valid_dataset, valid_loader = get_celeba(
            root=args.data_dir, split="valid", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
        test_dataset, test_loader = get_celeba(
            root=args.data_dir, split="test", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
    elif args.dataset == "utkface":
        train_dataset, train_loader = get_utkface(
            root=data_dir, split="train", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
        valid_dataset, valid_loader = get_utkface(
            root=data_dir, split="valid", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
        test_dataset, test_loader = get_utkface(
            root=data_dir, split="test", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
    elif args.dataset == "cifar10s":
        train_dataset, train_loader = get_cifar10s(root=data_dir, split="train", img_size=args.img_size, batch_size=args.batch_size)
        valid_dataset, valid_loader = get_cifar10s(root=data_dir, split="valid", img_size=args.img_size, batch_size=args.batch_size)
        test_dataset, test_loader = get_cifar10s(root=data_dir, split="test", img_size=args.img_size, batch_size=args.batch_size)

    # Load the pretrained models
    ckpt_path = os.path.join(save_dir, log_name_d + ".pt")
    deployed_model = ResNet18(num_classes=num_classes, pretrained=False).to(device)
    deployed_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    deployed_model.eval()

    # Test original test dataset
    num_correct = 0
    with torch.no_grad():
        all_labels, all_biases, all_preds = [], [], []
        for images, labels, biases in test_loader:
            images = images.to(device)

            logits, _ = deployed_model(images)
            preds = torch.argmax(logits.data, 1).cpu()
            num_correct += torch.sum(preds == labels)

            all_labels.append(labels)
            all_biases.append(biases)
            all_preds.append(preds)

        print("accuracy of original images in test dataset: %.3f\n" % (num_correct.item() / len(test_dataset)))

        fin_labels = torch.cat(all_labels)
        fin_biases = torch.cat(all_biases)
        fin_preds = torch.cat(all_preds)

        ret = get_all_metrics(y_true=fin_labels, y_pred=fin_preds, sensitive_features=fin_biases)
        print_all_metrics(ret=ret)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch: {epoch}")

        # Load the generators
        ckpt_path_G = os.path.join(save_dir, log_name_g + f"_epoch{epoch}.pt")
        print(f"Load checkpoint: {ckpt_path_G}")
        netG = Generator().to(device)
        netG.load_state_dict(torch.load(ckpt_path_G))
        netG.eval()

        # Test adversarial examples in test dataset
        num_correct = 0
        with torch.no_grad():
            all_labels, all_biases, all_preds = [], [], []
            for images, labels, biases in test_loader:
                images = images.to(device)
                perturbation = netG(images)
                perturbation = torch.clamp(perturbation, -0.05, 0.05)
                adv_img = perturbation + images
                if args.dataset == "celeba":
                    adv_img = torch.clamp(adv_img, 0, 1)
                elif args.dataset == "utkface":
                    adv_img = torch.clamp(adv_img, 0, 1)
                elif args.dataset == "cifar10s":
                    adv_img = torch.clamp(adv_img, -1, 1)

                logits, _ = deployed_model(adv_img)
                preds = torch.argmax(logits.data, 1).cpu()
                num_correct += torch.sum(preds == labels)

                all_labels.append(labels)
                all_biases.append(biases)
                all_preds.append(preds)

            print("accuracy of adversarial images in test dataset: %.3f\n" % (num_correct.item() / len(test_dataset)))

            fin_labels = torch.cat(all_labels)
            fin_biases = torch.cat(all_biases)
            fin_preds = torch.cat(all_preds)

            ret = get_all_metrics(y_true=fin_labels, y_pred=fin_preds, sensitive_features=fin_biases)
            print_all_metrics(ret=ret)


if __name__ == "__main__":
    test_adv(args)
