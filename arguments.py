import os
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Fairness")
    parser.add_argument("--rs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adversary-with-logits", action="store_true")

    parser.add_argument("--parallel", action="store_true", help="data parallel")
    parser.add_argument("--labelwise", action="store_true", help="labelwise loader")
    parser.add_argument("--jointfeature", action="store_true", help="mmd with both joint")
    parser.add_argument("--pretrained", action="store_true", help="load the pre-trained model")
    parser.add_argument("--checkpoint", action="store_true", help="load the trained model for testing")
    parser.add_argument("--no-annealing", action="store_true", help="do not anneal lamb during training")
    parser.add_argument("--get-inter", action="store_true", help="get penultimate features for TSNE visualization")
    parser.add_argument(
        "--adversary-with-y",
        action="store_true",
        help="True for Equalized Odds, target on [ED_PO1_AcrossZ], " "False for Demographic Parity, target on [ED_FR_AcrossZ].",
    )

    parser.add_argument("--bb", default=0, type=int)
    parser.add_argument("--uw", default=1, type=int)
    parser.add_argument("--aug", default=1, type=int)
    parser.add_argument("--ecu", default=0, type=int)
    parser.add_argument("--ratio", default=10, type=int)
    parser.add_argument("--wd", default=2e-5, type=float)
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--weight", default=0.01, type=float)
    parser.add_argument("--epochs_extra", default=20, type=int)
    parser.add_argument("--adversary-lr", default=1e-3, type=float)
    parser.add_argument("--gender_classifier", default="./bias_capturing_classifiers/bcc_gender.pth", type=str)
    parser.add_argument("--color_classifier", default="./bias_capturing_classifiers/bcc_cifar10s.pth", type=str)

    parser.add_argument("--lr", required=True, type=float, help="learning rate")
    parser.add_argument("--batch-size", required=True, type=int, help="mini batch size")
    parser.add_argument("--lr_layer", type=float, help="learning rate for the last layer")
    parser.add_argument("--epochs", required=True, type=int, help="number of training epochs")

    parser.add_argument("--gpu", default=0, type=int, help="CUDA visible device")
    parser.add_argument("--seed", default=1, type=int, help="seed for randomness")
    parser.add_argument("--save_freq", default=200, type=int, help="save frequency")
    parser.add_argument("--evalset", default="all", choices=["all", "train", "test"])
    parser.add_argument("--print_freq", default=300, type=int, help="print frequency")
    parser.add_argument("--kd-temp", default=3, type=float, help="temperature for KD")
    parser.add_argument("--sigma", default=1.0, type=float, help="sigma for rbf kernel")
    parser.add_argument("--optimizer", default="Adam", type=str, choices=["SGD", "Adam"])
    parser.add_argument("--constraint", default="EO", type=str, help="fairness constraint")
    parser.add_argument("--model-path", default=None, type=str, help="deployed model path")
    parser.add_argument("--teacher-path", default=None, type=str, help="teacher model path")
    parser.add_argument("--finetune-method", default="M2", type=str, help="finetune method")
    parser.add_argument("--lambh", default=4, type=float, help="kd strength hyperparameter")
    parser.add_argument("--img-size", default=224, type=int, help="image size for preprocessing")
    parser.add_argument("--skew-ratio", default=0.95, type=float, help="skew ratio for cifar-10s")
    parser.add_argument("--term", default=20, type=int, help="the period for recording train acc")
    parser.add_argument("--repeat-time", default=1, type=int, help="the number of experimental repeats")
    parser.add_argument("--lambf", default=1, type=float, help="feature distill strength hyperparameter")
    parser.add_argument("--cbs", default=64, type=int, help="batch_size of dataloader for contrastive loss")
    parser.add_argument("--training_ratio", default=2, type=float, help="training ratio for confusion loss")
    parser.add_argument("--num-workers", default=8, type=int, help="the number of thread used in dataloader")
    parser.add_argument("--target", default="Blond_Hair", type=str, help="target attribute for celeba dataset")
    parser.add_argument("--sensitive", default="race", type=str, help="sensitive attribute for utkface dataset")
    parser.add_argument("--lmbda", default=0.5, type=float, help="The coefficient of the adversarial loss applied to CE loss")
    parser.add_argument(
        "--reprogram-size",
        default=200,
        type=int,
        help="This parameter has different meanings for different reprogramming methods. "
        "For vanilla reprogramming method, the size of the resized image."
        "For reprogram patch, the patch size."
        "For optimization-based reprogram, the equivalent size of a patch for optimized pixels.",
    )
    parser.add_argument(
        "--trigger-data-num",
        default=0,
        type=int,
        help="How many data do you want to use to train reprogram, default for using the whole training set!",
    )

    parser.add_argument("--model", default="resnet18", type=str, choices=["resnet18"])
    parser.add_argument("--method", default="scratch", type=str, choices=["scratch", "kd_mfd"])
    parser.add_argument("--dataset", default="celeba", type=str, choices=["celeba", "utkface", "cifar10s"])
    parser.add_argument("--kernel", default="rbf", type=str, choices=["rbf", "poly"], help="kernel for mmd")
    parser.add_argument("--mode", default="none", type=str, help="mode for BM method", choices=["none", "us", "os", "uw"])
    parser.add_argument(
        "--rmethod",
        default="std",
        type=str,
        choices=["std", "adv", "repro", "rpatch", "roptim"],
        help="Method: standard training, adv training, reprogram (vanilla, patch, optimization-based)",
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    return args
