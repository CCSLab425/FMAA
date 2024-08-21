"""
Author:  Qi tong Chen
Date: 2024.08.20
Positive adversarial and negative adversarial
"""
import torch


def dcd_labels_calculate_positive(ground_truth_for_groups_8):
    # Groups containing the same category are aligned
    if ground_truth_for_groups_8 == 0:  # G2_1 or G5
        dcd_label_8 = 0  # G1_1 or G1
    elif ground_truth_for_groups_8 == 1:  # G2_2 or G6
        dcd_label_8 = 1  # G1_2 or G2
    elif ground_truth_for_groups_8 == 2:  # G2_3 or G7
        dcd_label_8 = 2  # G1_3 or G3
    elif ground_truth_for_groups_8 == 3:  # G2_4 or G8
        dcd_label_8 = 3  # G1_4 or G4
    elif ground_truth_for_groups_8 == 4:  # G4_1 or G13
        dcd_label_8 = 8  # G3_1 or G9
    elif ground_truth_for_groups_8 == 5:  # G4_2 or G14
        dcd_label_8 = 9  # G3_2 or G10
    elif ground_truth_for_groups_8 == 6:  # G4_3 or G15
        dcd_label_8 = 10  # G3_3 or G11
    elif ground_truth_for_groups_8 == 7:  # G4_4 or G16
        dcd_label_8 = 11  # G3_4 or G12

    return dcd_label_8


def dcd_labels_calculate_negative_1(ground_truth_for_groups_8):
    # Align groups containing different categories
    if ground_truth_for_groups_8 == 0:  # G2_1 or G5
        dcd_label_8_negative = 1  # G1_2 or G2
    elif ground_truth_for_groups_8 == 1:  # G2_2 or G6
        dcd_label_8_negative = 0  # G1_1 or G1
    elif ground_truth_for_groups_8 == 2:  # G2_3 or G7
        dcd_label_8_negative = 3  # G1_4 or G4
    elif ground_truth_for_groups_8 == 3:  # G2_4 or G8
        dcd_label_8_negative = 2  # G1_3 or G3
    elif ground_truth_for_groups_8 == 4:  # G4_1 or G13
        dcd_label_8_negative = 10  # G3_3 or G11
    elif ground_truth_for_groups_8 == 5:  # G4_2 or G14
        dcd_label_8_negative = 11  # G3_4 or G12
    elif ground_truth_for_groups_8 == 6:  # G4_3 or G15
        dcd_label_8_negative = 8  # G3_1 or G9
    elif ground_truth_for_groups_8 == 7:  # G4_4 or G16
        dcd_label_8_negative = 9  # G3_2 or G10
    return dcd_label_8_negative


def dcd_labels_calculate_negative_2(ground_truth_for_groups_8):
    # Align groups containing different categories
    if ground_truth_for_groups_8 == 0:  # G2_1 or G5
        dcd_label_8_negative2 = 2  # G1_3 or G3
    elif ground_truth_for_groups_8 == 1:  # G2_2 or G6
        dcd_label_8_negative2 = 2  # G1_3 or G3
    elif ground_truth_for_groups_8 == 2:  # G2_3 or G7
        dcd_label_8_negative2 = 0  # G1_1 or G1
    elif ground_truth_for_groups_8 == 3:  # G2_4 or G8
        dcd_label_8_negative2 = 0  # G1_1 or G1

    return dcd_label_8_negative2


def dcd_labels_calculate_negative_3(ground_truth_for_groups_8):
    # Align groups containing different categories
    if ground_truth_for_groups_8 == 0:  # G2_1 or G5
        dcd_label_8_negative3 = 3  # G1_4 or G4
    elif ground_truth_for_groups_8 == 1:  # G2_2 or G6
        dcd_label_8_negative3 = 3  # G1_4 or G4
    elif ground_truth_for_groups_8 == 2:  # G2_3 or G7
        dcd_label_8_negative3 = 1  # G1_2 or G2
    elif ground_truth_for_groups_8 == 3:  # G2_4 or G8
        dcd_label_8_negative3 = 1  # G1_2 or G2

    return dcd_label_8_negative3


def soft_max(net_output):
    X_exp = torch.exp(net_output)
    partition = X_exp.sum(1, keepdim=True)
    probability = X_exp / partition

    return probability

