import operator
import numpy as np
import matplotlib.pyplot as plt
import torch


def accuracy(net, dataloader, device, div=True, eval=True):
    net.to(device)
    if eval:
        net.eval()
    correct = 0
    class_accuracy = {}
    # print(dataloader.dataset.label_dict.keys())
    label_dct = dataloader.dataset.label_dict
    if div:
        label_dct = {k//2:label_dct[k] for k in label_dct}
    for label in label_dct:
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
                    if int(predicted.to(device)[idx]) in label_dct.keys():
                        class_accuracy[int(predicted.to(device)[idx])]['fPos'] += 1
                        class_accuracy[int(predicted.to(device)[idx])]['tNeg'] -= 1
                    class_accuracy[int(labels.to(device)[idx])]['fNeg'] += 1
                # add 1 to everyone true Negative except at `int(labels.to(device)[idx])`
                for label in label_dct.keys():
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


def plot_accuracy(class_accuracy, label_dict, decending=True):
    desired_stat = 'tPos'

    label_to_total = {k: [class_accuracy[k]['cls_size'],
                      class_accuracy[k][desired_stat]/class_accuracy[k]
                          ['cls_size']]
                      for k in class_accuracy.keys()}

    # get item at 0 to sort by name, get item at 1 to sort by accuracy
    sorted_tuples = sorted(label_to_total.items(), key=operator.itemgetter(0),
                           reverse=decending)
    sorted_dict = {k: v for k, v in sorted_tuples}

    width = 0.8
    plt.figure(figsize=(20, 7))  # width:20, height:10
    # [ label_dict[k]  for k in sorted_dict.keys()]
    plt.bar([label_dict[k*2] for k in sorted_dict.keys()],
            [val[1] for val in sorted_dict.values()], width, color='g',
            align='center')
    plt.xticks(rotation=90)
    plt.show()
    return sorted_dict


def get_confusion_matrix(class_num, net, test_loader, device, div=True,
                         eval=True):
    mat = np.zeros((class_num, class_num))
    net.to(device)
    if eval:
        net.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            if div:
                labels = labels // 2
            outputs = net(images.to(device))
            _, preds = torch.max(outputs.data, 1)
            assert len(labels) == len(preds), \
                "output & label shape unmatch"
            for i in range(len(labels)):
                label, pred = labels[i], preds[i]
                assert label < class_num and label >= 0
                assert pred < class_num and pred >= 0
                mat[label, pred] += 1
    return mat
