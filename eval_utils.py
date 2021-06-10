import copy
import operator
import numpy as np
import matplotlib.pyplot as plt
import torch


def get_class_type(n_arr):
    type_arr = []
    for n in n_arr:
        if n > 100:
            type_arr.append("many")
        elif n >= 20:
            type_arr.append("medium")
        else:
            type_arr.append("few")
    return type_arr


def data_dict_by_type(data_dict, type_arr):
    type_dict = {"many": {"annotations": [], "num_classes": 0},
                 "medium": {"annotations": [], "num_classes": 0},
                 "few": {"annotations": [], "num_classes": 0}}

    for class_dict in data_dict.values():
        data_arr = class_dict["annotations"]
        class_idx = data_arr[0]['category_id'] // 2
        num_classes = class_dict["num_classes"]
        type_dict[type_arr[class_idx]]["annotations"] += data_arr
        type_dict[type_arr[class_idx]]["num_classes"] += num_classes
    return type_dict


def accuracy_by_type(net, dataloader, device, div=True, eval=True):
    net.to(device)
    if eval:
        net.eval()
    accuracy_dict = {"many": 0, "medium": 0, "few": 0}



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


def classifier_simple_accuracy(net, classifier, dataloader, device, div=True,
                               eval=True):

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
            rep_out = net._dropout(net._avg_pooling(
                net.extract_features(images.to(device)))
                    .flatten(start_dim=1)).squeeze()
            outputs = classifier(rep_out.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return correct/total


def tau_sweep(net, tau_arr, device, full_loader, many_loader, 
              medium_loader=None, few_loader=None):
    weights = list(net._fc.parameters())[0].data.clone()
    normB = torch.norm(weights, 2, 1)

    for p in tau_arr:
        net.to(device)
        ws = weights.clone()

        for i in range(weights.size(0)):
            ws[i] = ws[i] / torch.pow(normB[i], p)
        fc = copy.deepcopy(net._fc)
        list(fc.parameters())[0].data = ws.to(device)
        list(fc.parameters())[1].data = torch.zeros(50).to(device)

        def classifier(rep):
            return net._swish(fc(rep))

        print(f"tau={p:.3f}\t", end="")
        overall_accu = classifier_simple_accuracy(net, classifier, full_loader,
                                                  device)
        print(f"overall accuracy: {overall_accu:.3f}\t", end="")
        many_accu = classifier_simple_accuracy(net, classifier, many_loader,
                                               device)
        print(f"many class accuracy: {many_accu:.3f}\t", end="")
        if medium_loader is not None:
            medium_accu = classifier_simple_accuracy(net, classifier,
                                                     medium_loader, device)
            print(f"medium class accuracy: {medium_accu:.3f}\t", end="")
        if few_loader is not None:
            few_accu = classifier_simple_accuracy(net, classifier,
                                                  few_loader, device)
            print(f"few class accuracy: {few_accu:.3f}\t", end="")
        print()


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
    plt.figure(figsize=(20, 7))  # width:20, height:7
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


def bagging_simple_accuracy(baggingnet, dataloader, device, div=True):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            if div:
                labels = labels//2
            predicted = baggingnet.predict(images.to(device))
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return correct/total
