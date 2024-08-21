"""
Author:  Qi tong Chen
Date: 2024.08.19
Training history and visualization functions
"""

from visdom import Visdom
import torch
import pandas as pd


# ======================================================================================================= #
#                                        Saving models and training history
# ======================================================================================================= #

def save_model(model=None, save_dir='save_dir/', model_name='defaults', source_data_name='', target_data_name=''):
    print('Save model!!!')
    model = model.cuda(0)
    save_path_and_name_1 = save_dir + '{}_{}_to_{}.pth'.format(
        model_name, source_data_name, target_data_name)
    save_path_and_name_2 = save_dir + "my_model_state_dict.pth"
    torch.save(model, save_path_and_name_1)  # Save the entire model, GPU mode
    torch.save(model.state_dict(), save_path_and_name_2)  # Only save the model's parameters


def save_his(train_hist={}, save_dir='save_dir/', save_name=''):
    """
    save history data
    """
    # for i in train_hist.keys():
    #     print('Head_name={}, values_length = {}'.format(i, len(train_hist[i])))
    data_df = pd.DataFrame(train_hist)
    data_df.to_csv(save_dir + save_name + '_history_results.csv')


def save_predict_labels(yt_label=None, yt_pre_label=None, save_dir='save_dir/', save_name=''):
    """
    save best predict labels
    :param yt_label:
    :param yt_pre_label:
    :param save_dir: save path
    :param save_name:
    :return:
    """
    best_prediction_labels = {}
    best_prediction_labels['yt_pre_label'] = []
    best_prediction_labels['yt_label'] = []
    yt_label = yt_label.cpu().detach().data.numpy()
    yt_pre_label_new = yt_pre_label.cpu().detach().data.numpy()
    best_prediction_labels['yt_label'] = yt_label
    best_prediction_labels['yt_pre_label'] = yt_pre_label_new
    prediction_lab = pd.DataFrame(best_prediction_labels)
    prediction_lab.to_csv(save_dir + save_name + '_prediction_labels.csv')


def train_history():
    train_hist = {}

    train_hist['sour_and_tar_mmd_loss_cls_1'] = []
    train_hist['sour_and_tar_mmd_loss_cls_2'] = []
    train_hist['sour_and_tar_mmd_loss_cls_3'] = []
    train_hist['sour_and_tar_mmd_loss_cls_4'] = []

    train_hist['Sour_test_loss'] = []
    train_hist['Sour_test_accuracy'] = []
    train_hist['Tar_test_loss'] = []
    train_hist['Tar_test_accuracy'] = []

    return train_hist