import os
import sys
import torch
from pathlib import Path
from numpy import mean, std

project_dir = "/root/DL-Fairness-Study"
sys.path.insert(1, os.path.join(project_dir, "FAAP"))

from models import Generator, ResNet18
from datasets.celeba import get_celeba
from datasets.utkface import get_utkface
from datasets.cifar10s import get_cifar10s

sys.path.insert(1, project_dir)
from helper import set_seed
from arguments import get_args
from metrics import get_metric_index, get_all_metrics, print_all_metrics

args = get_args()


def test_adv(args):
    print(args)
    set_seed(args.seed)

    if args.dataset in ["celeba", "cifar10s"]:
        exp_name = f"faap-{args.dataset}-lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-seed{args.seed}"
    elif args.dataset == "utkface":
        exp_name = f"faap-utkface_{args.sensitive}-lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-seed{args.seed}"
    save_dir = Path(f"{project_dir}/checkpoints/{exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

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

    data_dir = f"{project_dir}/data"
    log_name_g = f"{project_dir}/checkpoints/{exp_name}"
    print(f"log name of generator: {log_name_g}")
    log_name_d = f"{project_dir}/checkpoints/{args.model_path}"
    print(f"log name of deployed model: {log_name_d}")
    num_classes = 10 if args.dataset == "cifar10s" else 2

    # Load datasets
    dataset_dir = f"{data_dir}/{args.dataset}"
    if args.dataset == "celeba":
        train_dataset, train_loader = get_celeba(
            root=data_dir, split="train", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
        valid_dataset, valid_loader = get_celeba(
            root=data_dir, split="valid", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
        test_dataset, test_loader = get_celeba(
            root=data_dir, split="test", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
    elif args.dataset == "utkface":
        train_dataset, train_loader = get_utkface(
            root=dataset_dir, split="train", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
        valid_dataset, valid_loader = get_utkface(
            root=dataset_dir, split="valid", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
        test_dataset, test_loader = get_utkface(
            root=dataset_dir, split="test", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
    elif args.dataset == "cifar10s":
        train_dataset, train_loader = get_cifar10s(root=dataset_dir, split="train", img_size=args.img_size, batch_size=args.batch_size)
        valid_dataset, valid_loader = get_cifar10s(root=dataset_dir, split="valid", img_size=args.img_size, batch_size=args.batch_size)
        test_dataset, test_loader = get_cifar10s(root=dataset_dir, split="test", img_size=args.img_size, batch_size=args.batch_size)

    # Load the pretrained models
    ckpt_path = f"{log_name_d}/best_model.pt"
    print(ckpt_path)
    deployed_model = ResNet18(num_classes=num_classes, pretrained=False).cuda()
    deployed_model.load_state_dict(torch.load(ckpt_path))
    deployed_model.eval()

    # Test original test dataset
    num_correct = 0
    with torch.no_grad():
        all_labels, all_biases, all_preds = [], [], []
        for images, labels, biases in test_loader:
            images = images.cuda()

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

    # Load the generators
    ckpt_path_G = f"{log_name_g}/best_model.pt"
    print(f"Load checkpoint: {ckpt_path_G}")
    netG = Generator().cuda()
    netG.load_state_dict(torch.load(ckpt_path_G))
    netG.eval()

    # Test adversarial examples in test dataset
    num_correct = 0
    with torch.no_grad():
        all_labels, all_biases, all_preds = [], [], []
        for images, labels, biases in test_loader:
            images = images.cuda()
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
