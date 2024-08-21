"""
Author:  Qi tong Chen
Date: 2024.08.19
Loading data and grouping with few-shots
"""

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from os.path import splitext
import scipy
import os
from torch.utils.data import DataLoader, TensorDataset
from files_path import files_path
import scipy.io as io


def data_reader(datadir, gpu=True):
    """
    read data from mat or other file
    Args:
        datadir: The name of the data file to be loaded
    """
    datatype = splitext(datadir)[1]
    if datatype == '.mat':
        data = scipy.io.loadmat(datadir)
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']

    if datatype == '':
        pass
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    y_train = torch.argmax(y_train, 1)  # dim=1, get the maximum value of the row
    y_test = torch.argmax(y_test, 1)
    return x_train, y_train, x_test, y_test


def source_dataload(batch_size=64, source_data_name=''):
    """
    Loading source domain dataset
    :param batch_size:
    :param source_data_name:
    :return:
    """
    s_dataset_path, _, = files_path()
    sdatadir = s_dataset_path + source_data_name + '.mat'
    xs_train, ys_train, xs_test, ys_test = data_reader(sdatadir)
    torch_dataset = TensorDataset(xs_test, ys_test)
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader


def target_dataload(batch_size=64, target_data_name=''):
    """
    Loading target domain dataset
    :param batch_size:
    :param target_data_name:
    :return:
    """
    _, t_dataset_path = files_path()
    tdatadir = t_dataset_path + target_data_name + '.mat'
    xt_train, yt_train, xt_test, yt_test = data_reader(tdatadir)
    torch_dataset = TensorDataset(xt_test, yt_test)
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader


def source_and_target_loader(batch_size=64, source_data_name='', target_data_name=''):
    """
    Loading source domain and target domain datasets
    :param batch_size:
    :param source_data_name:
    :param target_data_name:
    :return:
    """
    s_dataset_path, t_dataset_path = files_path()
    sdatadir = s_dataset_path + source_data_name + '.mat'
    tdatadir = t_dataset_path + target_data_name + '.mat'
    xs_train, ys_train, xs_test, ys_test = data_reader(sdatadir)
    xt_train, yt_train, xt_test, yt_test = data_reader(tdatadir)

    torch_dataset = TensorDataset(xs_test, ys_test, xt_test, yt_test)
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader


def source_sample(source_data_name=''):
    """
    Randomly sample from the source domain samples and form a new dataset (disrupting the order of the original dataset)
    :return:
    """
    s_dataset_path, _ = files_path()
    sdatadir = s_dataset_path + source_data_name + '.mat'
    xs_train, ys_train, xs_test, ys_test = data_reader(sdatadir)

    source_dataset = TensorDataset(xs_test, ys_test)
    n = len(source_dataset)
    X = torch.Tensor(n, 1, 32, 32)  # Initialization
    Y = torch.LongTensor(n)  # Initialization

    inds = torch.randperm(len(source_dataset))  # generate random index

    for i, index in enumerate(inds):
        x, y = source_dataset[index]
        X[i] = x
        Y[i] = y
    return X, Y


def target_sample(n=1, target_data_name=''):
    """
    Sampling few-shots from the target domain dataset
    :param n: the number of samples in per class
    :param target_data_name:
    :return:
    """
    _, t_dataset_path, = files_path()
    tdatadir = t_dataset_path + target_data_name + '.mat'
    xt_train, yt_train, xt_test, yt_test = data_reader(tdatadir)

    target_dataset = TensorDataset(xt_train, yt_train)
    X, Y = [], []
    classes = 4 * [n]  # Each class is taken n times, classes = [n, n, n, n]

    i = 0
    while True:
        if len(X) == n * 4:
            break  # It ends when n items are taken from each category.
        x, y = target_dataset[i]  # y.dtype = torch.int64
        if classes[y] > 0:   # Take n (shot) times for each category.
            y_float = y.type(torch.float32)
            X.append(x)
            Y.append(y)
            # Y.append(y_float)
            classes[y] -= 1  # Indicates that the category has been taken once
        i += 1
    assert (len(X) == n * 4)
    return torch.stack(X, dim=0), torch.from_numpy(np.array(Y))


def create_groups(X_s, Y_s, X_t, Y_t, seed=1):
    # change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)
    # =======================================================================================================
    #      Determine the value of shot and disrupt the independent and non-repeating class sequence
    # =======================================================================================================
    n = X_t.shape[0]  # classes * shot
    # shuffle order
    classes = torch.unique(Y_t)  # Extract independent and non-repetitive class labels, indicating the total number of classes
    # print('unique classes = ', classes)
    classes = classes[torch.randperm(len(classes))]

    class_num = classes.shape[0]
    shot = n // class_num

    # =======================================================================================================
    #   Randomly sample shot*2 samples from each class in the source domain,
    #   and randomly sample shot samples from each class in the created target domain samples to generate idx
    # =======================================================================================================
    def s_idxs(c):
        """
        First, find the index of the same class as C in the source domain based on the independent
        and non-repetitive label C of the target domain, then randomly shuffle the (400) idxs,
        and finally take the first shot*2 and compress them into a one-dimensional tensor.
        :param c:
        :return:
        """
        idx = torch.nonzero(Y_s.eq(int(c)))
        return idx[torch.randperm(len(idx))][:shot*2].squeeze()

    def t_idxs(c):
        """
        Find the label of the same category as C in the target domain based on the independent
        and non-repetitive labels of the target domain.
        :param c:The target domain contains independent and non-repetitive labels.
        :return:
        """
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix = torch.stack(source_idxs)  # [class, 2*shot]
    target_matrix = torch.stack(target_idxs)  # [class, shot]
    if shot == 1:   # Only a small number of samples of each class in the target domain can be used
        # When shot=1, target_matrix is 1-dimensional and needs to be upgraded.
        target_matrix = target_matrix.reshape(target_matrix.shape[0], -1)
    G1_1, G1_2, G1_3, G1_4 = [], [], [], []
    Y1_1, Y1_2, Y1_3, Y1_4 = [], [], [], []

    G2_1, G2_2, G2_3, G2_4 = [], [], [], []
    Y2_1, Y2_2, Y2_3, Y2_4 = [], [], [], []

    G3_1, G3_2, G3_3, G3_4 = [], [], [], []
    Y3_1, Y3_2, Y3_3, Y3_4 = [], [], [], []

    G4_1, G4_2, G4_3, G4_4 = [], [], [], []
    Y4_1, Y4_2, Y4_3, Y4_4 = [], [], [], []
    for i in range(4):
        for j in range(shot):

            if i == 0:
                G1_1.append((X_s[source_matrix[i][j * 2]], X_s[source_matrix[i][j * 2 + 1]]))
                Y1_1.append((Y_s[source_matrix[i][j * 2]], Y_s[source_matrix[i][j * 2 + 1]]))
                G2_1.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
                Y2_1.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))
                G3_1.append((X_s[source_matrix[i % class_num][j]], X_s[source_matrix[(i + 1) % class_num][j]]))
                Y3_1.append((Y_s[source_matrix[i % class_num][j]], Y_s[source_matrix[(i + 1) % class_num][j]]))
                G4_1.append((X_s[source_matrix[i % class_num][j]], X_t[target_matrix[(i + 1) % class_num][j]]))
                Y4_1.append((Y_s[source_matrix[i % class_num][j]], Y_t[target_matrix[(i + 1) % class_num][j]]))
            elif i == 1:
                G1_2.append((X_s[source_matrix[i][j * 2]], X_s[source_matrix[i][j * 2 + 1]]))
                Y1_2.append((Y_s[source_matrix[i][j * 2]], Y_s[source_matrix[i][j * 2 + 1]]))
                G2_2.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
                Y2_2.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))
                G3_2.append((X_s[source_matrix[i % class_num][j]], X_s[source_matrix[(i + 1) % class_num][j]]))
                Y3_2.append((Y_s[source_matrix[i % class_num][j]], Y_s[source_matrix[(i + 1) % class_num][j]]))
                G4_2.append((X_s[source_matrix[i % class_num][j]], X_t[target_matrix[(i + 1) % class_num][j]]))
                Y4_2.append((Y_s[source_matrix[i % class_num][j]], Y_t[target_matrix[(i + 1) % class_num][j]]))
            elif i == 2:
                G1_3.append((X_s[source_matrix[i][j * 2]], X_s[source_matrix[i][j * 2 + 1]]))
                Y1_3.append((Y_s[source_matrix[i][j * 2]], Y_s[source_matrix[i][j * 2 + 1]]))
                G2_3.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
                Y2_3.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))
                G3_3.append((X_s[source_matrix[i % class_num][j]], X_s[source_matrix[(i + 1) % class_num][j]]))
                Y3_3.append((Y_s[source_matrix[i % class_num][j]], Y_s[source_matrix[(i + 1) % class_num][j]]))
                G4_3.append((X_s[source_matrix[i % class_num][j]], X_t[target_matrix[(i + 1) % class_num][j]]))
                Y4_3.append((Y_s[source_matrix[i % class_num][j]], Y_t[target_matrix[(i + 1) % class_num][j]]))
            elif i == 3:
                G1_4.append((X_s[source_matrix[i][j * 2]], X_s[source_matrix[i][j * 2 + 1]]))
                Y1_4.append((Y_s[source_matrix[i][j * 2]], Y_s[source_matrix[i][j * 2 + 1]]))
                G2_4.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
                Y2_4.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))
                G3_4.append((X_s[source_matrix[i % class_num][j]], X_s[source_matrix[(i + 1) % class_num][j]]))
                Y3_4.append((Y_s[source_matrix[i % class_num][j]], Y_s[source_matrix[(i + 1) % class_num][j]]))
                G4_4.append((X_s[source_matrix[i % class_num][j]], X_t[target_matrix[(i + 1) % class_num][j]]))
                Y4_4.append((Y_s[source_matrix[i % class_num][j]], Y_t[target_matrix[(i + 1) % class_num][j]]))

    groups_16 = [G1_1, G1_2, G1_3, G1_4,
                 G2_1, G2_2, G2_3, G2_4,
                 G3_1, G3_2, G3_3, G3_4,
                 G4_1, G4_2, G4_3, G4_4]
    groups_16_y = [Y1_1, Y1_2, Y1_3, Y1_4,
                   Y2_1, Y2_2, Y2_3, Y2_4,
                   Y3_1, Y3_2, Y3_3, Y3_4,
                   Y4_1, Y4_2, Y4_3, Y4_4]
    return groups_16, groups_16_y


def sample_groups(X_s, Y_s, X_t, Y_t, seed=1):

    print("Sampling groups")
    return create_groups(X_s, Y_s, X_t, Y_t, seed=seed)


if __name__ == '__main__':
    X_s, Y_s = source_sample(source_data_name='CWRU_1hp_4')  # xs = [4*400, 1, 28, 28], ys = [4*400]
    print('X_s.shape, Y_s.shape = ', X_s.shape, Y_s.shape)
    X_t, Y_t = target_sample(n=1, target_data_name='SBDS_1K_4_06')  # xs=[classes*shot, 1, 32, 32], ys=[classes*shot]
    print('X_t.shape, Y_t.shape = ', X_t.shape, Y_t.shape)

    groups_16, groups_16_y = sample_groups(X_s, Y_s, X_t, Y_t, seed=0)
    source_dataload(batch_size=64, source_data_name='CWRU_1hp_4')
    print('G1_1={}, G1_2={}, G1_3={}, G1_4={}'.format(groups_16_y[0], groups_16_y[1], groups_16_y[2], groups_16_y[3]))
    print('G2_1={}, G2_2={}, G2_3={}, G2_4={}'.format(groups_16_y[4], groups_16_y[5], groups_16_y[6], groups_16_y[7]))
    print('G3_1={}, G3_2={}, G3_3={}, G3_4={}'.format(groups_16_y[8], groups_16_y[9], groups_16_y[10], groups_16_y[11]))
    print(
        'G4_1={}, G4_2={}, G4_3={}, G4_4={}'.format(groups_16_y[12], groups_16_y[13], groups_16_y[14], groups_16_y[15]))

