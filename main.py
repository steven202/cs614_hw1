import argparse
import glob
import numpy as np
from pkg_resources import packaging
import os
import pprint
import torch
import torchvision
import tqdm

from glob import glob
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import load_txt, accuracy, create_barplot, get_fname, AverageMeter
from models.resnet import ResNet56
from dataset import CIFAR10C
import clip
from sklearn.metrics import confusion_matrix

CORRUPTIONS = load_txt("./corruptions.txt")
MEAN = [0.49139968, 0.48215841, 0.44653091]
STD = [0.24703223, 0.24348513, 0.26158784]


# code is modified from the following website:
# reference: https://vitalflux.com/accuracy-precision-recall-f1-score-python-example/
def plot_cm(conf_matrix, cname=""):
    plt.close()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large")

    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("Actuals", fontsize=18)
    # ax.xaxis.set_ticks(list(range(1, 11)))
    # ax.yaxis.set_ticks(list(range(1, 11)))
    # plt.yticks(labels=list(range(1, 11)))
    plt.locator_params(nbins=10)
    # ax.locator_params(nbins=10, axis="x")
    # ax.locator_params(nbins=10, axis="y")
    plt.title(f"Confusion Matrix of {cname}", fontsize=18)
    plt.show()
    plt.savefig(f"confusion_matrix_{cname}.png", dpi=300, bbox_inches="tight")


def main(opt, weight_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model
    model, preprocess = clip.load("ViT-B/32")
    model.to(device).eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print(clip.available_models())
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    accs = dict()
    classes_ = datasets.CIFAR10(
        os.path.join(opt.data_root, "cifar10"),
        train=False,
        transform=transform,
        download=True,
    ).classes
    text_descriptions = [f"This is a photo of a {label}" for label in classes_]
    text_tokens = clip.tokenize(text_descriptions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    with tqdm(total=len(opt.corruptions), ncols=80) as pbar:
        for ci, cname in enumerate(opt.corruptions):
            # load dataset
            if cname == "natural":
                dataset = datasets.CIFAR10(
                    os.path.join(opt.data_root, "cifar10"),
                    train=False,
                    transform=preprocess,
                    download=True,
                )
                # continue
            else:
                dataset = CIFAR10C(os.path.join(opt.data_root, "CIFAR-10-C"), cname, transform=preprocess)
            loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

            acc_meter = AverageMeter()
            with torch.no_grad():
                for itr, (x, y) in enumerate(loader):
                    x = x.to(device, non_blocking=True)
                    with torch.no_grad():
                        image_features = model.encode_image(x).float()
                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    top_probs, top_labels = text_probs.cpu().topk(10, dim=-1)
                    y = y.to(device, dtype=torch.int64, non_blocking=True)

                    # z = model(x)
                    # z = top_labels.squeeze(dim=1).to(device)
                    z = top_probs.to(device)
                    # loss = F.cross_entropy(z, y)
                    # acc, _ = accuracy(z, y, topk=(1,))
                    acc = (
                        (text_probs.cpu().topk(1, dim=-1)[1].squeeze(dim=1).to(device) == y).sum() / y.numel() * 100.0
                    )
                    # print(cname, acc.item())
                    if not opt.no_print_cm:
                        cm = confusion_matrix(
                            y.cpu(), text_probs.cpu().topk(1, dim=-1)[1].squeeze(dim=1).to(device).cpu()
                        )
                        plot_cm(cm, cname)
                    acc_meter.update(acc.item())

            accs[f"{cname}"] = acc_meter.avg

            pbar.set_postfix_str(f"{cname}: {acc_meter.avg:.2f}")
            pbar.update()

    avg = np.mean(list(accs.values()))
    accs["avg"] = avg
    # accs = {k: v * 100.0 for k, v in accs.items()}
    pprint.pprint(accs)
    # save_name = get_fname(weight_path)
    save_name = "Accuracy vs Corruption"
    # create_barplot(accs, save_name + f" / avg={avg:.2f}", os.path.join(opt.fig_dir, save_name + ".png"))
    create_barplot(accs, save_name + f", Average Accuracy={avg:.2f} %", os.path.join("", save_name + ".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="resnet18", help="model name")
    parser.add_argument(
        "--weight_dir",
        type=str,
        help="path to the dicrectory containing model weights",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        help="path to the dicrectory containing model weights",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="figs",
        help="path to the dicrectory saving output figure",
    )
    parser.add_argument("--data_root", type=str, default="~/data", help="root path to cifar10-c directory")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        nargs="*",
        default=CORRUPTIONS,
        help="testing corruption types",
    )
    parser.add_argument(
        "--no_print_cm",
        # type=bool,
        default=False,
        action="store_true",
        help="print confusion matrix",
    )
    parser.add_argument("--gpu_id", type=str, default=0, help="gpu id to use")

    opt = parser.parse_args()
    main(opt, None)
