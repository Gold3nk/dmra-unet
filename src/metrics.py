import numpy as np
import torch

# 实验用
from scipy.spatial.distance import cdist
from medpy.metric.binary import hd, hd95


# 训练指标计算
class MetricsPt(object):
    def __init__(self):
        """Training/Validation Metrics"""
        self._EPS = 1e-7

    # 输入为张量
    def accuracy(self, y_pred, y_true):
        """
        y_pred: (B, H, W, D)    # from argmax
        y_true: (B, H, W, D)    # no channel dimension
        """
        return torch.sum(y_pred == y_true).float() / (1. * y_true.nelement())

    # 输入为二值化体积张量
    def dice_score(self, y_pred_bin, y_true_bin):
        """
        y_pred_bin: (B, H, W, D)    # binary volume
        y_true_bin: (B, H, W, D)    # binary volume
        预测的二值化体积
        真实的二值化体积
        """
        return (2.0 * torch.sum(y_true_bin * y_pred_bin) + self._EPS) / \
               (torch.sum(y_true_bin) + torch.sum(y_pred_bin) + self._EPS)

    # 输入为张量
    def get_dice_per_region(self, y_pred, y_true):
        """
        y_pred: (B, H, W, D)    # from argmax
        y_true: (B, H, W, D)    # no channel dimension
        """
        # 将 y_pred 张量中大于0的元素置为 True，小于等于0的元素置为 False，返回一个布尔类型的张量。
        # 然后 .float() 是将布尔类型的张量转换为浮点类型的张量，其中 True 被转换为1.0， False 被转换为0.0。
        # 此操作不会改变 y_pred 本身。
        y_pred_wt = torch.gt(y_pred, 0).float()
        y_true_wt = torch.gt(y_true, 0).float()
        dice_wt = self.dice_score(y_pred_wt, y_true_wt)

        # (torch.eq(y_pred, 1) | torch.eq(y_pred, 3)) 是一个逻辑运算，用于判断 y_pred 张量中的元素是否等于1或等于3，返回一个布尔类型的张量。
        # 其中 True 表示相应位置的元素满足条件，而 False 表示不满足条件。
        y_pred_tc = (torch.eq(y_pred, 1) | torch.eq(y_pred, 3)).float()
        y_true_tc = (torch.eq(y_true, 1) | torch.eq(y_true, 3)).float()
        dice_tc = self.dice_score(y_pred_tc, y_true_tc)

        y_pred_en = torch.eq(y_pred, 3).float()
        y_true_en = torch.eq(y_true, 3).float()
        dice_en = self.dice_score(y_pred_en, y_true_en)

        return dice_wt, dice_tc, dice_en


# 预测指标计算
class MetricsNp(object):
    def __init__(self):
        """Test Time Metrics"""
        self._EPSILON = 1e-7

    def __call__(self, data, label):
        """"""
        return self.get_dice_per_region(data, label)

    def dice_score_binary_np(self, data, label):
        """
        f1-score <or> dice score: 2TP / (2TP + FP + FN)
        Args:
            label: 'np.array' binary volume.
            data: 'np.array' binary volume.
        Returns:
            dice_score (scalar)
        """
        numerator = 2 * np.sum(label * data)
        denominator = np.sum(label + data)
        # 分子部分是2倍的TP，表示预测结果和真实标签的交集的大小。
        # 分母部分是2倍的TP加上FP和FN的总和，表示预测结果和真实标签的并集的大小。
        score = (numerator + self._EPSILON) / (denominator + self._EPSILON)
        if np.isnan(score).any() or np.isinf(score).any():
            return 'NA'
        return score

    def get_dice_per_region(self, data, label):  # data = Prediction, label = Ground Truth
        """
        Provides region-wise Dice scores of a multi-class prediction simultaneously
        """
        assert data.shape == label.shape  # Shape check
        print('> Calculating Metrics ...')
        unique_labels = np.unique(label)
        num_classes = len(unique_labels)
        print('Total Classes:\t', num_classes)
        print('Class Labels:\t', unique_labels)

        # Whole Tumor:
        wt_data = np.float32(data > 0)
        wt_label = np.float32(label > 0)
        wt_score = self.dice_score_binary_np(wt_data, wt_label)
        # 这样处理后，wt_data中的像素值为0和1，其中1表示预测结果中属于整个肿瘤区域的像素。

        # Tumor Core:
        tc_data = np.float32((data == 1) | (data == 4))
        tc_label = np.float32((label == 1) | (label == 4))
        tc_score = self.dice_score_binary_np(tc_data, tc_label)
        # 逻辑或操作的结果是，如果两个操作数中至少有一个为真（非零），则结果为真（非零）。
        # 因此，对于tc_data矩阵，如果像素值等于1或等于4，则对应位置的值为1，其他位置的值为0。

        # Enhancing Core:
        en_data = np.float32(data == 4)
        en_label = np.float32(label == 4)
        en_score = self.dice_score_binary_np(en_data, en_label)

        return wt_score, tc_score, en_score

    # 实验用
    # def Dice(self, output, target, eps=1e-3):
    #     inter = torch.sum(output * target, dim=(1, 2, 3)) + eps
    #     union = torch.sum(output, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + eps * 2
    #     x = 2 * inter / union
    #     dice = torch.mean(x)
    #     return dice
    #
    # def cal_dice(self, output, target):
    #     output = torch.argmax(output, dim=1)
    #     dice1 = self.Dice((output == 3).float(), (target == 3).float())
    #     dice2 = self.Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    #     dice3 = self.Dice((output != 0).float(), (target != 0).float())
    #
    #     return dice1, dice2, dice3

    def hausdorff_distance_binary_up(self, data, label):
        # if len(np.unique(label)) < 2:
        #     print('Skip sample with insufficient label.')
        #     return None

        hausdorff_distance = hd(data, label)

        # # 计算95%豪斯多夫距离
        # hausdorff_distance95 = hd95(predict, ground_truth)

        return hausdorff_distance

    def get_hausdorff_per_region(self, data, label):  # data = Prediction, label = Ground Truth
        """
        Provides region-wise Dice scores of a multi-class prediction simultaneously
        """
        assert data.shape == label.shape  # Shape check
        print('> Calculating Metrics ...')
        unique_labels = np.unique(label)
        if len(unique_labels) < 4:
            print('Skip sample with insufficient label.')
            return None, None, None
        num_classes = len(unique_labels)
        print('Total Classes:\t', num_classes)
        print('Class Labels:\t', unique_labels)

        # Whole Tumor:
        wt_data = np.float32(data > 0)
        wt_label = np.float32(label > 0)
        wt_score = self.hausdorff_distance_binary_up(wt_data, wt_label)

        # Tumor Core:
        tc_data = np.float32((data == 1) | (data == 4))
        tc_label = np.float32((label == 1) | (label == 4))
        tc_score = self.hausdorff_distance_binary_up(tc_data, tc_label)

        # Enhancing Core:
        en_data = np.float32(data == 4)
        en_label = np.float32(label == 4)
        en_score = self.hausdorff_distance_binary_up(en_data, en_label)

        return wt_score, tc_score, en_score

    def hausdorff95_distance_binary_up(self, data, label):
        # if len(np.unique(label)) < 2:
        #     print('Skip sample with insufficient label.')
        #     return None

        # hausdorff_distance = hd(data, label)

        # 计算95%豪斯多夫距离
        hausdorff95_distance = hd95(data, label)

        return hausdorff95_distance

    def get_hausdorff95_per_region(self, data, label):  # data = Prediction, label = Ground Truth
        """
        Provides region-wise Dice scores of a multi-class prediction simultaneously
        """
        assert data.shape == label.shape  # Shape check
        print('> Calculating Metrics ...')
        unique_labels = np.unique(label)
        # 如果标签 (0, 1, 2, 4) 少了，就跳过。
        if len(unique_labels) < 4:
            print('Skip sample with insufficient label.')
            return None, None, None
        num_classes = len(unique_labels)
        print('Total Classes:\t', num_classes)
        print('Class Labels:\t', unique_labels)

        # Whole Tumor:
        wt_data = np.float32(data > 0)
        wt_label = np.float32(label > 0)
        wt_score = self.hausdorff95_distance_binary_up(wt_data, wt_label)

        # Tumor Core:
        tc_data = np.float32((data == 1) | (data == 4))
        tc_label = np.float32((label == 1) | (label == 4))
        tc_score = self.hausdorff95_distance_binary_up(tc_data, tc_label)

        # Enhancing Core:
        en_data = np.float32(data == 4)
        en_label = np.float32(label == 4)
        en_score = self.hausdorff95_distance_binary_up(en_data, en_label)

        return wt_score, tc_score, en_score


if __name__ == '__main__':
    print('No *elaborate* testing routine implemented')  # 打印：未实现复杂的测试例程
    metrics_obj = MetricsPt()
    x = torch.zeros(240, 240, 155)
    y = torch.zeros(240, 240, 155)
    print(metrics_obj.dice_score(x, y))
