import os, sys
from os.path import dirname, realpath
import torch
import tqdm.notebook as tq
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim
import numpy as np

# LS >>> what is this, winston?
sys.path.append(dirname(realpath(__file__))) 

from sam.sam import SAM
from eval_utils import simple_accuracy


def get_SGD(model_params, lr, momentum, decay):
    return optim.SGD(model_params, lr=lr, momentum=momentum,
                     weight_decay=decay)


def get_Adam(model_params, lr, momentum, decay):
    return optim.Adam(model_params, lr=lr, weight_decay=decay)


def get_SAM(model_params, lr, momentum, decay):
    base_optimizer = torch.optim.SGD
    return SAM(model_params, base_optimizer, lr=lr, momentum=momentum,
               weight_decay=decay)

def get_lossWeights(beta, num_classes, data_dict):
    # >>> begin getting weights

    # get dictionary of training data [now expected as parameter]

    # convert to dictionary of quantity of training data per class
    label_to_quant = {key//2:len(data_dict[key]['annotations']) for key in data_dict}

    # compute weights based on
    # https://arxiv.org/pdf/1901.05555.pdf
    # https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab
    effective_num = np.zeros(num_classes)
    for i in label_to_quant:
        effective_num[i] = label_to_quant[i]

    effective_num = 1 - np.power(beta, effective_num)
    weights = (1-beta)/effective_num
    weights = weights/np.sum(weights) * num_classes
    weights = torch.FloatTensor(weights)
    return weights


def get_model(device, num_classes, net_str, optim_type, model=None, lr=0.001,
              momentum=0.9, decay=0.0005, loss_weight=None):
    """optim_type == SGD | Adam | SAM"""
    hidden_count_dict = {
        "efficientnet-b0": 1280,
        "efficientnet-b1": 1280,
        "efficientnet-b2": 1408,
        "efficientnet-b3": 1536,
        "efficientnet-b4": 1792,
        "efficientnet-b5": 2048,
        "efficientnet-b6": 2304,
        "efficientnet-b7": 2560,
        "efficientnet-b8": 2816,
        "efficientnet-l2": 5504
    }
    if model is None:
        model = EfficientNet.from_name(net_str)
        model._fc = nn.Linear(hidden_count_dict[net_str], num_classes)
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optim_dict = {"SGD": get_SGD, "Adam": get_Adam, "SAM": get_SAM}
    assert optim_type in optim_dict, "invalid optim_type"
    optimizer = optim_dict[optim_type](model.parameters(), lr, momentum, decay)
    model.to(device)
    criterion.to(device)

    return model, criterion, optimizer


def train(checkpoint_dir, net, train_loader, vali_loader, net_str, data_label,
          rho, device, criterion, optimizer, beta=0.0, epochs=1):
    losses = []
    vali_losses = []
    vali_accues = []

    # write save dir
    if net_str == "efficientnet-b0":
        net_str = "EfficientNet"
    loss_str = type(criterion).__name__
    optim_str = type(optimizer).__name__
    lr = optimizer.param_groups[0]['lr']
    gamma = optimizer.param_groups[0]['weight_decay']
    assert lr is not None and gamma is not None, "optimizer parameter error"
    print(f"training params: net={net_str}\tloss={loss_str}\toptim={optim_str}"
          f"\tepochs={epochs}\tlr={lr}\tweight_decay={float(gamma)}")

    def url_func(epo):
        return os.path.join(
            checkpoint_dir,
            f"{net_str}_{loss_str}_beta{beta}_{optim_str}_lr{lr}_gamma{gamma}_"
            f"{data_label}_rho{rho}_epoch{epo}.pt")

    # load in saved checkpoints
    start_epoch = 0
    for i in range(epochs):
        curr_epoch = epochs-i
        curr_url = url_func(curr_epoch)
        if os.path.exists(curr_url):
            print(f"load checkpoint from={curr_url}")
            state = torch.load(curr_url)
            net.load_state_dict(state['net'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch']
            vali_losses = state['vali_losses']
            vali_accues = state['vali_accues']
            break

    epoch = start_epoch
    print("File Path:", url_func(start_epoch))
    for epoch in range(start_epoch, epochs):
        net.train()
        t = tq.tqdm(enumerate(train_loader), desc=f"train epoch={epoch}",
                    position=0, leave=True, total=len(train_loader))
        for i, batch in t:
            inputs, labels = batch

            def closure():
                loss = criterion(net(inputs.to(device)),
                                 (labels//2).to(device))
                loss.backward()
                return loss
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            # ***map to network output, specific for CIFAR-50***
            loss = criterion(net(inputs.to(device)), (labels//2).to(device))
            loss.backward()
            if optim_str == "SAM":
                optimizer.step(closure)
            else:
                optimizer.step()
            losses.append(loss.item())
            t.set_description(f"loss={loss.item():.3f}\tepoch={epoch}")

        # validation
        net.eval()
        vali_accuracy = simple_accuracy(net, vali_loader, device)
        print("epoch {}\tloss={:.3f}\taccuracy={:.3f}".format(
            epoch, losses[-1], vali_accuracy))
        vali_losses.append(losses[-1])
        vali_accues.append(vali_accuracy)

        # save state_dict
        state = {'epoch': epoch+1, 'net': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'vali_losses': vali_losses,
                 'vali_accues': vali_accues}
        curr_save_url = url_func(epoch+1)
        prev_save_url = url_func(epoch)
        print(f"save model checkpoint at={curr_save_url}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, curr_save_url)
        if os.path.exists(prev_save_url) and epoch % 10 != 0:
            os.remove(prev_save_url)

    # net.load_state_dict(best_state['net'])
    return net, vali_losses, vali_accues


def load_cRT_model(root_dir, device, net_str, loss_str, optim_str, rho, lr,
                   gamma, beta, epochs, num_classes, optim_type, data_label, 
                   cRT=True, resampled=True):
    checkpoint_dir = os.path.join(root_dir, "checkpoints")
    hidden_count_dict = {
        "efficientnet-b0": 1280,
        "efficientnet-b1": 1280,
        "efficientnet-b2": 1408,
        "efficientnet-b3": 1536,
        "efficientnet-b4": 1792,
        "efficientnet-b5": 2048,
        "efficientnet-b6": 2304,
        "efficientnet-b7": 2560,
        "efficientnet-b8": 2816,
        "efficientnet-l2": 5504
    }
    if net_str == "efficientnet-b0":
        cp_net_str = "EfficientNet"
    else:
        cp_net_str = net_str
    model_base = f"{cp_net_str}_{loss_str}_beta{beta}_{optim_str}_lr{lr}_gamma{gamma}_" \
                 f"{data_label}_rho{rho}_epoch{epochs}"
    model_url = os.path.join(checkpoint_dir, f"{model_base}.pt")
    if not os.path.exists(model_url):
        print(f"model url: {model_url} doesn't exist")
        return
    print(f"load model from={model_url}")
    print(f"model params: net={net_str}\tloss={loss_str}\toptim={optim_str}"
          f"\tepochs={epochs}\tlr={lr}\tweight_decay={float(gamma)}")
    state = torch.load(model_url)
    model = EfficientNet.from_name(net_str)
    model._fc = nn.Linear(hidden_count_dict[net_str], num_classes)
    model.load_state_dict(state['net'])

    if resampled:
        cRT_folder = os.path.join(checkpoint_dir, f"{model_base}_cRT")
    else:
        cRT_folder = os.path.join(
            checkpoint_dir, f"{model_base}_resampled0_cRT")
    if cRT:
        if not os.path.exists(cRT_folder):
            os.makedirs(cRT_folder)

        # fix representation & randomize classifier
        for param in model.parameters():
            param.requires_grad = False
        model._fc = nn.Linear(hidden_count_dict[net_str], num_classes) \
            .to(device)
        model._dropout = nn.Dropout(p=0.2, inplace=False).to(device)
        model._avg_pooling = nn.AdaptiveAvgPool2d(output_size=1).to(device)
        model._bn1 = nn.BatchNorm2d(
            hidden_count_dict[net_str], eps=0.001,
            momentum=0.010000000000000009,
            affine=True, track_running_stats=True).to(device)
        model.train()
    return model, cRT_folder
