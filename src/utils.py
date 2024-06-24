import os
import pathlib
import pprint

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import yaml

import matplotlib  # 实验用
matplotlib.use('Agg')  # 实验用
from matplotlib import pyplot as plt
"""
use('Agg')这里是因为matplotlib的默认backend是TkAgg
而FltkAgg, GTK, GTKAgg, GTKCairo, TkAgg , Wx or WxAgg这几个backend都要求有GUI图形界面的
但是linux环境是没有图形界面的
所以需要手动指定为不需要GUI的backend--------Agg, Cairo, PS, PDF or SVG
"""

from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff

from medpy.metric.binary import hd, hd95  # 实验用
# hausdorff_distance95=hd95(predict,ground_truth)  # 示例

from torch import distributed as dist
from torch.cuda.amp import autocast

from src.dataset.batch_utils import pad_batch1_to_compatible_size


def save_args(args):
    config = vars(args).copy()
    del config['save_folder']
    del config['seg_folder']
    pprint.pprint(config)
    config_file = args.save_folder / (args.exp_name + ".yaml")
    with config_file.open("w") as file:
        yaml.dump(config, file)


def master_do(func, *args, **kwargs):
    try:
        rank = dist.get_rank()
        if rank == 0:
            return func(*args, **kwargs)
    except AssertionError:
        # not in DDP setting, just do as usual
        func(*args, **kwargs)


def save_checkpoint(state: dict, save_folder: pathlib.Path):
    best_filename = f'{str(save_folder)}/model_best.pth.tar'
    torch.save(state, best_filename)


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# TODO remove dependency to args
def reload_ckpt(args, model, optimizer, scheduler):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))


def reload_ckpt_bis(ckpt, model, optimizer=None):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        try:
            checkpoint = torch.load(ckpt)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{ckpt}' (epoch {start_epoch})")
            return start_epoch
        except RuntimeError:
            # TO account for checkpoint from Alex nets
            print("Loading model Alex style")
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 计算评估指标
# 它接收预测结果（preds）、目标结果（targets）、患者ID（patient）和是否进行测试时间增强（tta）作为输入。
# 将Hausdorff距离、Dice系数、灵敏度和特异性等评估指标添加到字典中，并将字典打印出来。同时，将字典添加到指标列表中。完成所有类别的遍历后，返回指标列表。
def calculate_metrics(preds, targets, patient, tta=False):
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):  # 遍历每个类别。
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        # # 保存.nii.gz图像
        # for i in [0, 1, 2]:
        #     output_folder = r'D:\Subject\code\open_brats2020-main\datasets\brats2021'
        #     targetsss = targets[i].astype(np.uint8)
        #     sitk_img = sitk.GetImageFromArray(targetsss)
        #     sitk.WriteImage(sitk_img, output_folder + f'/targetsss[{i}].nii.gz')
        #     predsss = preds[i].astype(np.uint8)
        #     sitk_img = sitk.GetImageFromArray(predsss)
        #     sitk.WriteImage(sitk_img, output_folder + f'/predsss[{i}].nii.gz')

        if np.sum(targets[i]) == 0:  # 如果目标结果中该类别的像素总数为0，则表示该类别在该病人中不存在。
            print(f"{label} not present for {patient}")
            sens = np.nan  # 灵敏度（sensitivity）设置为NaN。
            dice = 1 if np.sum(preds[i]) == 0 else 0
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            spec = tn / (tn + fp)  # Specificity
            haussdorf_dist = np.nan
            # haussdorf95_dist = np.nan  # 实验用
            prec = np.nan  # 实验用 precision 精确率

        else:
            # np.argwhere函数用于获取非零元素的坐标，即找到数组中值为True的位置。
            # preds_coords和targets_coords分别存储了预测结果和目标结果中非零元素的坐标。
            preds_coords = np.argwhere(preds[i])
            targets_coords = np.argwhere(targets[i])
            haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]
            # haussdorf95_dist = directed_hausdorff(preds_coords, targets_coords)[0]  # 新增
            # haussdorf95_dist = hd95(preds_coords, targets_coords)  # 实验新增

            tp = np.sum(l_and(preds[i], targets[i]))
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            prec = tp / (tp + fp)  # 实验用

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[HAUSSDORF] = haussdorf_dist

        metrics[DICE] = dice
        metrics[SENS] = sens
        metrics[SPEC] = spec
        metrics[PREC] = prec  # 实验用
        # metrics[HAUSSDORF95] = haussdorf95_dist  # 实验用

        pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list


class WeightSWA(object):

    def __init__(self, swa_model):
        self.num_params = 0
        self.swa_model = swa_model  # assume that the parameters are to be discarded at the first update

    def update(self, student_model):
        self.num_params += 1
        print("Updating SWA. Current num_params =", self.num_params)
        if self.num_params == 1:
            print("Loading State Dict")
            self.swa_model.load_state_dict(student_model.state_dict())
        else:
            inv = 1. / float(self.num_params)
            for swa_p, src_p in zip(self.swa_model.parameters(), student_model.parameters()):
                swa_p.data.add_(-inv * swa_p.data)
                swa_p.data.add_(inv * src_p.data)

    def reset(self):
        self.num_params = 0


def save_metrics(epoch, metrics, swa, writer, current_epoch, teacher=False, save_folder=None):
    metrics = list(zip(*metrics))
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    labels = ("ET", "TC", "WT")
    metrics = {key: value for key, value in zip(labels, metrics)}
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics.values(), labels=metrics.keys())
    ax.set_ylim(0, 1)
    writer.add_figure(f"val/plot", fig, global_step=epoch)
    print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
              [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()], file=f)
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{'_swa' if swa else ''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)


# 生成 results.csv
def generate_segmentations(data_loader, model, writer, args):
    metrics_list = []
    for i, batch in enumerate(data_loader):
        # measure data loading time
        inputs = batch["image"]
        patient_id = batch["patient_id"][0]
        ref_path = batch["seg_path"][0]
        crops_idx = batch["crop_indexes"]
        inputs, pads = pad_batch1_to_compatible_size(inputs)
        inputs = inputs.cuda()
        with autocast():
            with torch.no_grad():
                if model.deep_supervision:
                    pre_segs, _ = model(inputs)
                else:
                    pre_segs = model(inputs)
                pre_segs = torch.sigmoid(pre_segs)
        # remove pads
        # 去除填充
        maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
        pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()

        segs = torch.zeros((1, 3, 155, 240, 240))
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
        segs = segs[0].numpy() > 0.5  # seg[0] 这一操作去除了 batch size 维度。此后 seg.shape 为 (4, 155, 240, 240)。为布尔类型。
        # 此即为最终分割结果，下列labelmap相关步骤是为了制作分割图。

        # segs[0] 增强肿瘤
        # segs[1] 肿瘤核心（增强肿瘤 + 坏死）
        # segs[2] 整个肿瘤（增强肿瘤 + 坏死 + 水肿）
        et = segs[0]  # 增强肿瘤
        net = np.logical_and(segs[1], np.logical_not(et))  # 坏死。  既属于segs[1]又不属于segs[0]的区域。
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))  # 水肿。  既属于segs[2]又不属于segs[1]的区域。
        bg = np.logical_not(segs[2])  # 背景区域。  不属于segs[2]的区域。

        labelmap = np.zeros(segs[0].shape)  # (155, 240, 240)

        # labelmap[et] = 4 这行代码的作用是将 et 区域对应的位置在 labelmap 中赋值为 4，表示增强肿瘤区域。
        # 在这行代码中，et 是一个布尔类型的数组，它的形状与 labelmap 相同。布尔类型的数组可以用作索引，其中 True 的位置会被选择出来。
        # 通过使用 et 数组作为索引，我们可以将 labelmap 中与 et 对应位置的元素赋值为 4。这意味着在最终的标签图像 labelmap 中，与增强肿瘤区域重叠的位置将被标记为 4。
        labelmap[et] = 4  # 增强肿瘤区域
        labelmap[net] = 1  # 坏死区域
        labelmap[ed] = 2  # 水肿区域
        labelmap[bg] = 0  # 水肿区域
        labelmap = sitk.GetImageFromArray(labelmap)  # 分割结果图

        # 获取真值标签
        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)
        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        # 将ref_seg中等于4的元素设为True，其余元素设为False，并将结果保存到refmap_et数组中
        refmap_et = ref_seg == 4  # 增强肿瘤
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)  # 肿瘤核心（增强肿瘤 + 坏死）
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)  # 整个肿瘤（增强肿瘤 + 坏死 + 水肿）
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])  # 真值标签图的数组

        # 计算结果指标
        patient_metric_list = calculate_metrics(segs, refmap, patient_id)  # (3, 155, 240, 240) (3, 155, 240, 240)
        metrics_list.append(patient_metric_list)  # 添加到列表里

        # 将labelmap与ref_seg_img的空间信息进行对齐
        # 将labelmap的空间信息、方向、像素大小等属性拷贝到ref_seg_img中
        labelmap.CopyInformation(ref_seg_img)

        print(f"Writing {args.seg_folder}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{args.seg_folder}/{patient_id}.nii.gz")  # 将labelmap保存为NIfTI格式的文件

    # 保存结果指标
    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    overlap = df.boxplot(METRICS[1:], by="label", return_type="axes")
    overlap_figure = overlap[0].get_figure()
    writer.add_figure("benchmark/overlap_measures", overlap_figure)
    haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    writer.add_figure("benchmark/distance_measure", haussdorf_figure)
    grouped_df = df.groupby("label")[METRICS]
    summary = grouped_df.mean().to_dict()
    for metric, label_values in summary.items():
        for label, score in label_values.items():
            writer.add_scalar(f"benchmark_{metric}/{label}", score)
    df.to_csv((args.save_folder / 'results.csv'), index=False)


def update_teacher_parameters(model, teacher_model, global_step, alpha=0.99 / 0.999):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        teacher_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    # print("teacher updated!")


HAUSSDORF = "haussdorf"  # 单词拼写错误

DICE = "dice"
SENS = "sens"
SPEC = "spec"
PREC = "prec"  # 实验用

METRICS = [HAUSSDORF, DICE, SENS, SPEC, PREC]  # 实验用
