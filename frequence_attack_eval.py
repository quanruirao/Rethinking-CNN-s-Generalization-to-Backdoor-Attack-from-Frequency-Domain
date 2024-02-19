import copy
import os
import random

import config
from frequence_change import DCT
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18, VGG, MobileNetV2
from networks.models import Denormalizer, Normalizer
from torch import nn
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from fractions import Fraction

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        # netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
        # netC = VGG("VGG16", opt.num_classes).to(opt.device)
        # netC = MobileNetV2(opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18(num_classes=opt.num_classes, data_name="celeba").to(opt.device)
        # netC = VGG("VGG16", opt.num_classes, is_celeba=True).to(opt.device)
        # netC = MobileNetV2(opt.num_classes, is_celeba=True).to(opt.device)
    if opt.dataset == "mnist":
        netC = ResNet18(num_classes=opt.num_classes, in_channel=1).to(opt.device)
        # netC = VGG("VGG16", opt.num_classes, in_channel=1).to(opt.device)
        # netC = MobileNetV2(opt.num_classes, in_channel=1).to(opt.device)


    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)
    # optimizerC = torch.optim.Adam(netC.parameters(), opt.lr_C, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    opt,
    tearget_test_datasets,):

    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)

            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            rd = random.randint(0, len(tearget_test_datasets)-1)

            if opt.dataset == "mnist":
                is_mnist = True
            else:
                is_mnist = False

            inputs_bd = DCT(copy.deepcopy(tearget_test_datasets[rd]), copy.deepcopy(inputs),
                            img_size=opt.input_height,dct_size=int(inputs.shape[2] * opt.dctsize),is_mnist=is_mnist,intensity=opt.intensity)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label

            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            info_string = "Clean Acc: {:.4f}| Bd Acc: {:.4f}".format(
                acc_clean, acc_bd
            )

            progress_bar(batch_idx, len(test_dl), info_string)



def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    opt.dctsize = float(Fraction(opt.dctsize))

    test_dl,transforms = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))

    if os.path.exists(opt.ckpt_path):
        state_dict = torch.load(opt.ckpt_path)
        netC.load_state_dict(state_dict["netC"])
    else:
        print("Pretrained model doesnt exist")
        exit()



    target_test_images = []

    for target_test_batch_idx, (target_test_data, target_test_targets) in enumerate(test_dl):

        target_test_indices = torch.nonzero(target_test_targets == opt.target_label).squeeze()

        if target_test_indices.numel() > 0:

            target_test_batch_images = target_test_data[target_test_indices]
            target_test_images.append(target_test_batch_images)

    target_test_images = torch.cat(target_test_images, dim=0)
    tearget_test_datasets = target_test_images.data

    eval(
        netC=netC,
        optimizerC=optimizerC,
        schedulerC=schedulerC,
        test_dl=test_dl,
        opt=opt,
        tearget_test_datasets= tearget_test_datasets,
    )


if __name__ == "__main__":
    main()
