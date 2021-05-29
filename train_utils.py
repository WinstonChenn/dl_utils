import os, sys
from os.path import dirname, realpath
import torch
import tqdm.notebook as tq
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim

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


def get_model(device, num_classes, optim_type, lr=0.001, momentum=0.9,
              decay=0.0005, loss_weight=None):
    """optim_type == SGD | Adam | SAM"""
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = nn.Linear(1280, num_classes)
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optim_dict = {"SGD": get_SGD, "Adam": get_Adam, "SAM": get_SAM}
    assert optim_type in optim_dict, "invalid optim_type"
    optimizer = optim_dict[optim_type](model.parameters(), lr, momentum, decay)
    model.to(device)
    criterion.to(device)

    return model, criterion, optimizer


def train(checkpoint_dir, net, train_loader, vali_loader, data_label, rho,
          device, criterion, optimizer, epochs=1, verbose=1, print_every=10):
    losses = []
    vali_losses = []
    vali_accues = []

    # write save dir
    net_str = type(net).__name__
    loss_str = type(criterion).__name__ 
    if type(criterion) == nn.CrossEntropyLoss:
        loss_str += "_0" if criterion.__dict__['_buffers']['weight'] == None else "_1"
    optim_str = type(optimizer).__name__
    lr = optimizer.param_groups[0]['lr']
    gamma = optimizer.param_groups[0]['weight_decay']
    assert lr is not None and gamma is not None, "optimizer parameter error"
    print(f"training params: net={net_str}\tloss={loss_str}\toptim={optim_str}"
          f"\tepochs={epochs}\tlr={lr}\tweight_decay={float(gamma)}")

    def url_func(epo):
        return os.path.join(
            checkpoint_dir,
            f"{net_str}_{loss_str}_{optim_str}_lr{lr}_gamma{gamma}_"
            f"{data_label}_rho{rho}_epoch{epo}.pt")
    checkpoint_url = url_func(epochs)

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
            t.set_description(f"loss={loss.item():.3f}\tepoch={epoch:.3f}")

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
        torch.save(state, curr_save_url)
        if os.path.exists(prev_save_url) and epoch % 10 != 0:
            os.remove(prev_save_url)

    # net.load_state_dict(best_state['net'])
    return net, vali_losses, vali_accues
