import numpy as np
import matplotlib as plt
import torch


def accuracy(net, dataloader, device, div=True, eval=True):
    net.to(device)
    if eval:
        net.eval()
    correct = 0
    class_accuracy = {}
    # print(dataloader.dataset.label_dict.keys())
    for label in dataloader.dataset.label_dict.keys():
        class_accuracy[label] = {'tPos': 0, 'fPos': 0, 'tNeg': 0,
                                 'fNeg': 0, 'cls_size': 0}
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            if div:
                labels = labels//2
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            for idx in range(labels.size(0)): 
                if labels.to(device)[idx] == predicted[idx]:
                    class_accuracy[int(labels.to(device)[idx])]['tPos'] += 1
                else:
                    if int(predicted.to(device)[idx]) in dataloader.dataset.label_dict.keys():
                        class_accuracy[int(predicted.to(device)[idx])]['fPos'] += 1
                        class_accuracy[int(predicted.to(device)[idx])]['tNeg'] -= 1
                    class_accuracy[int(labels.to(device)[idx])]['fNeg'] += 1
                # add 1 to everyone true Negative except at `int(labels.to(device)[idx])`
                for label in dataloader.dataset.label_dict.keys():
                    class_accuracy[label]['tNeg'] += 1  # add one to everyone's true negative
                class_accuracy[int(labels.to(device)[idx])]['tNeg'] -= 1
                class_accuracy[int(labels.to(device)[idx])]['cls_size'] += 1

    return correct/total, class_accuracy


# used by training. can't get label dictionary for concat dataset
def simple_accuracy(net, dataloader, device, div=True, eval=True):
    net.to(device)
    if eval:
        net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            if div:
                labels = labels//2
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return correct/total


def smooth(x, size=50):
    return np.convolve(x, np.ones(size)/size, mode='valid')


def plot_losses(losses):
    plt.plot(smooth(losses, 50))


def plot_vali_stat(loss_arr, accu_arr):
    fig, axes = plt.subplots(1, 2, figsize=[6.4*2, 4.8])
    axes[0].set_title("losses")
    axes[0].plot(loss_arr)
    axes[1].set_title("accurcy")
    axes[1].plot(accu_arr)