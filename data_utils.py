import os, operator, random, wget, tarfile
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import ConcatDataset
import numpy as np


# dataset object
class Cifar50Dataset(Dataset):

    def __init__(self, cifar_json, transform, label_type_arr=None):
        self.class_num = cifar_json['num_classes']
        self.annotations = cifar_json['annotations']
        self.transform = transform
        self.label_type_arr = label_type_arr

        self.label_dict = {}
        for idx, data_ref in enumerate(self.annotations):
            label = data_ref['category']
            label_id = data_ref['category_id']
            if label_id not in self.label_dict:
                self.label_dict[label_id] = label

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations[idx]['fpath']
        label_id = self.annotations[idx]['category_id']
        
        if self.label_type_arr is not None:
            assert label_id % 2 == 0
            label_id = self.label_type_arr[label_id // 2]

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        return image, label_id

def class2class_type(class_n_arr, num_classes):
    class_type_arr = class_n_arr.copy()
    assert len(class_n_arr) == num_classes
    for i in range(num_classes):
        if class_n_arr[i] > 100:
            class_type_arr[i] = 0
        elif class_n_arr[i] >= 20:
            class_type_arr[i] = 1
        else:
            class_type_arr[i] = 2
    return class_type_arr


# utils for original cifar100 dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def arr2tensor(arr):
    assert len(arr) == 3072
    return torch.tensor([arr[:1024].reshape(32, 32),
                        arr[1024:2048].reshape(32, 32),
                        arr[2048:3072].reshape(32, 32)])


def get_cifar100_labels():
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = os.path.basename(url)
    if not os.path.exists(filename):
        filename = wget.download(url)
    assert filename.endswith("tar.gz"), "tar file url error"
    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()
    basename = os.path.splitext(os.path.splitext(filename)[0])[0]
    meta = unpickle(os.path.join(basename, "meta"))
    fine_labels = [n.decode("utf-8") for n in meta[b'fine_label_names']]
    coarse_labels = [n.decode("utf-8") for n in meta[b'coarse_label_names']]

    return fine_labels, coarse_labels


def print_random_cifar100_test(fine_labels, coarse_labels):
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = os.path.basename(url)
    basename = os.path.splitext(os.path.splitext(filename)[0])[0]
    test = unpickle(os.path.join(basename, "test"))
    rand_idx = random.randint(0, len(test[b'data'])-1)
    fine_label = fine_labels[test[b'fine_labels'][rand_idx]]
    coarse_label = coarse_labels[test[b'coarse_labels'][rand_idx]]
    img = arr2tensor(test[b'data'][rand_idx])
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"{fine_label} {coarse_label}")


# Other dataset utils
def print_random_img(dataset, label_map, n=3):
    fig, axes = plt.subplots(1, n, figsize=(8, 8*n))
    for i in range(n):
        rand_idx = random.randint(0, len(dataset))
        img, target = dataset[rand_idx]
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f"{label_map[target]} {target}")


def print_random_img_by_dataset(dataset_arr, label_map, n=3):
    rand_idx_arr = [random.randint(0, len(dataset_arr[0])) for i in range(n)]
    for idx, dataset in enumerate(dataset_arr):
        fig, axes = plt.subplots(1, n, figsize=(8, 8*n))
        for i in range(n):
            img, target = dataset[rand_idx_arr[i]]
            axes[i].imshow(img.permute(1, 2, 0))
            axes[i].set_title(f"{label_map[target]} {target}")
        fig.show()


def print_random_augmented_img(aug_dataset, original_size, label_map):
    assert len(aug_dataset) % original_size == 0
    n = len(aug_dataset) // original_size
    rand_idx = random.randint(0, original_size-1)
    fig, axes = plt.subplots(1, n, figsize=(8, 8*n))
    for i in range(n):
        img, target = aug_dataset[rand_idx+i*original_size]
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(label_map[target])


# data augmentation utils
def data_augmentation_X8(data_json, label_type_arr=None):
    """perform a data augmentation that expands dataset size by 7"""
    tt_transform = transforms.Compose([transforms.ToTensor()])
    rh_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                       transforms.ToTensor()])
    rv_transform = transforms.Compose([transforms.RandomVerticalFlip(p=1),
                                       transforms.ToTensor()])
    rr_transform = transforms.Compose([
        transforms.RandomRotation(90, expand=False, center=None,
                                  interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor()])
    ra_transform = transforms.Compose([
        transforms.RandomAffine(90, translate=(0.2, 0.2), scale=(0.9, 1.1)),
        transforms.ToTensor()])
    cj_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5,
                               contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor()])
    rp_transform = transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.5, p=1, fill=0,
                                     interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor()])
    re_transform = transforms.Compose([
        transforms.ToTensor(), transforms.RandomErasing(p=1)])

    dataset_og = Cifar50Dataset(data_json, tt_transform, label_type_arr)
    dataset_rh = Cifar50Dataset(data_json, rh_transform, label_type_arr)
    dataset_rv = Cifar50Dataset(data_json, rv_transform, label_type_arr)
    dataset_ra = Cifar50Dataset(data_json, ra_transform, label_type_arr)
    dataset_rr = Cifar50Dataset(data_json, rr_transform, label_type_arr)
    dataset_cj = Cifar50Dataset(data_json, cj_transform, label_type_arr)
    dataset_rp = Cifar50Dataset(data_json, rp_transform, label_type_arr)
    dataset_re = Cifar50Dataset(data_json, re_transform, label_type_arr)

    dataset_arr = [dataset_og, dataset_rh, dataset_rv, dataset_ra, dataset_rr,
                   dataset_cj, dataset_rp, dataset_re]
    aug_dataset = ConcatDataset(dataset_arr)

    return aug_dataset, dataset_arr


def generate_jsonDict(main_json):
    data_ref_arr = main_json["annotations"]
    class_num = main_json["num_classes"]
    json_dict = {}
    label_dict = {}

    for idx, data_ref in enumerate(data_ref_arr):
        label = data_ref['category']
        label_id = data_ref['category_id']
        if label_id not in label_dict:
            label_dict[label_id] = label
            json_dict[label_id] = {"annotations": [], "num_classes": 1}
        json_dict[label_id]["annotations"].append(data_ref)

    return json_dict, label_dict


def get_resample_prob_by_class(n_arr, class_num, q):
    assert class_num == len(n_arr)
    denom = np.sum(np.power(n_arr, q))
    prob_arr = np.power(n_arr, q)/denom

    return prob_arr


def get_resample_prob_by_instance(dataset, aug_factor, q, n_arr, class_num):
    ori_p = []
    class_p = get_resample_prob_by_class(n_arr, class_num, q)
    for _, label in dataset:
        ori_p.append(class_p[label//2])
    p_arr = ori_p * aug_factor
    return p_arr


def get_class_balanced_weights(dataset, aug_factor, n_arr, class_num):
    class_p = get_resample_prob_by_class(n_arr, class_num, q=0)
    ori_p = []
    for _, label in dataset:
        ori_p.append(class_p[label//2]/n_arr[label//2])
    p_arr = ori_p * aug_factor
    return np.array(p_arr)/sum(p_arr)


def plot_distribution(json_dict, label_dict, decending=True):
    label_to_length = {k: len(json_dict[k]['annotations'])
                       for k in json_dict.keys()}

    sorted_tuples = sorted(label_to_length.items(),
                           key=operator.itemgetter(1),
                           reverse=decending)
    sorted_dict = {k: v for k, v in sorted_tuples}

    width = 1.0
    plt.figure(figsize=(20, 7))  # width:20, height:10
    # [ label_dict[k]  for k in sorted_dict.keys()]
    plt.bar([label_dict[k] for k in sorted_dict.keys()],
            sorted_dict.values(), width, color='g', align='center')
    plt.xticks(rotation=90)
    plt.show()
    return sorted_dict
