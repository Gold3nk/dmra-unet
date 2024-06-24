
import argparse
import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace

import SimpleITK as sitk
import numpy as np
import pandas as pd

import torch
import torch.optim
import torch.utils.data
import yaml

from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from src import models
from src.dataset import get_datasets
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from src.models import get_norm_layer
from src.tta import apply_simple_tta
from src.utils import reload_ckpt_bis, calculate_metrics
from src.config import BRATS_TEST_FOLDER


# torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='Brats validation and testing dataset inference')
parser.add_argument('--config', default=[r'D:\xxx\xxx\xxx.yaml'], type=str, metavar='PATH', nargs="+")
parser.add_argument('--devices', required=True, type=str)
parser.add_argument('--on', default="test", choices=["val", "train", "test"])
parser.add_argument('--tta', action="store_true")
parser.add_argument('--seed', default=16111990)


def main(args):
    # setup
    random.seed(args.seed)
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeWarning("This will not be able to run on CPU only")
    print(f"Working with {ngpus} GPUs")
    print(args.config)

    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    args.save_folder = pathlib.Path(f"./preds/{current_experiment_time}")
    args.save_folder.mkdir(parents=True, exist_ok=True)

    with (args.save_folder / 'args.txt').open('w') as f:
        print(vars(args), file=f)

    args_list = []
    for config in args.config:

        config_file = pathlib.Path(config).resolve()
        # config_file = pathlib.Path(config)  # 不额外转化为绝对路径

        print(config_file)
        ckpt = config_file.with_name("model_best.pth.tar")
        with config_file.open("r") as file:
            old_args = yaml.safe_load(file)
            old_args = SimpleNamespace(**old_args, ckpt=ckpt)
            # set default normalisation
            if not hasattr(old_args, "normalisation"):
                old_args.normalisation = "minmax"
        print(old_args)
        args_list.append(old_args)

    if args.on == "test":
        args.pred_folder = args.save_folder / f"test_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    elif args.on == "val":
        args.pred_folder = args.save_folder / f"validation_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    else:
        args.pred_folder = args.save_folder / f"training_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)

    # Create model

    models_list = []
    normalisations_list = []
    for model_args in args_list:
        print(model_args.arch)
        model_maker = getattr(models, model_args.arch)

        model = model_maker(
            4, 3,
            width=model_args.width, deep_supervision=model_args.deep_sup,
            norm_layer=get_norm_layer(model_args.norm_layer), dropout=model_args.dropout)
        print(f"Creating {model_args.arch}")

        reload_ckpt_bis(str(model_args.ckpt), model)
        models_list.append(model)
        normalisations_list.append(model_args.normalisation)
        print("reload best weights")
        print(model)

    dataset_minmax = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="minmax")

    dataset_zscore = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="zscore")

    loader_minmax = torch.utils.data.DataLoader(
        dataset_minmax, batch_size=1, num_workers=2)

    loader_zscore = torch.utils.data.DataLoader(
        dataset_zscore, batch_size=1, num_workers=2)

    print("Val dataset number of batch:", len(loader_minmax))
    # print(normalisations_list)  # ['minmax']
    generate_segmentations((loader_minmax, loader_zscore), models_list, normalisations_list, args)


# data_loaders列表中包含 两种不同标准化方法获得的 数据
def generate_segmentations(data_loaders, models, normalisations, args):
    # TODO: try reuse the function used for train...
    metrics_list = []
    for i, (batch_minmax, batch_zscore) in enumerate(zip(data_loaders[0], data_loaders[1])):
        patient_id = batch_minmax["patient_id"][0]

        # ref_img_path = batch_minmax["seg_path"][0]
        ref_img_path = os.path.join(BRATS_TEST_FOLDER, patient_id, patient_id + '_seg.nii.gz')

        crops_idx_minmax = batch_minmax["crop_indexes"]
        crops_idx_zscore = batch_zscore["crop_indexes"]
        inputs_minmax = batch_minmax["image"]
        inputs_zscore = batch_zscore["image"]
        inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
        inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
        model_preds = []
        last_norm = None
        for model, normalisation in zip(models, normalisations):
            if normalisation == last_norm:
                pass
            elif normalisation == "minmax":
                inputs = inputs_minmax.cuda()
                pads = pads_minmax
                crops_idx = crops_idx_minmax
            elif normalisation == "zscore":
                inputs = inputs_zscore.cuda()
                pads = pads_zscore
                crops_idx = crops_idx_zscore
            model.cuda()  # go to gpu
            with autocast():
                with torch.no_grad():
                    if args.tta:
                        pre_segs = apply_simple_tta(model, inputs, True)
                        model_preds.append(pre_segs)
                    else:
                        if model.deep_supervision:
                            pre_segs, _ = model(inputs)
                        else:
                            pre_segs = model(inputs)
                        pre_segs = pre_segs.sigmoid_().cpu()
                    # remove pads
                    # 去除填充
                    maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - \
                                       pads[2]
                    pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
                    print("pre_segs size", pre_segs.shape)

                    segs = torch.zeros((1, 3, 155, 240, 240))
                    segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
                    print("segs size", segs.shape)

                    model_preds.append(segs)
            model.cpu()  # free for the next one

        pre_segs = torch.stack(model_preds).mean(dim=0)  # [1, 3, 155, 240, 240]
        segs = pre_segs[0].numpy() > 0.5  # (3, 155, 240, 240)

        # 制作并保存图像
        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        bg = np.logical_not(segs[2])
        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap[bg] = 0
        labelmap = sitk.GetImageFromArray(labelmap)
        ref_img = sitk.ReadImage(ref_img_path)
        # print('这里', sitk.GetArrayFromImage(ref_img), '和这里', ref_img_path)  # 实验用
        labelmap.CopyInformation(ref_img)
        print(f"Writing {str(args.pred_folder)}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{str(args.pred_folder)}/{patient_id}.nii.gz")  # 将分割图像保存为NIfTI格式的文件

        # 获取真值标签
        ref_seg_img = sitk.ReadImage(ref_img_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)
        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        refmap_et = ref_seg == 4
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])  # 真值标签图的数组 (3, 155, 240, 240)

        # 计算结果指标
        patient_metric_list = calculate_metrics(segs, refmap, patient_id)  # (3, 155, 240, 240) (3, 155, 240, 240)
        metrics_list.append(patient_metric_list)  # 添加到列表里

    # 保存结果指标
    # 定义声明
    HAUSSDORF = "haussdorf"
    DICE = "dice"
    SENS = "sens"
    SPEC = "spec"
    METRICS = [HAUSSDORF, DICE, SENS, SPEC]
    tt_writer = SummaryWriter(str(args.save_folder))

    test_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(test_metrics)
    overlap = df.boxplot(METRICS[1:], by="label", return_type="axes")
    overlap_figure = overlap[0].get_figure()
    tt_writer.add_figure("benchmark/overlap_measures", overlap_figure)
    haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    tt_writer.add_figure("benchmark/distance_measure", haussdorf_figure)
    grouped_df = df.groupby("label")[METRICS]
    summary = grouped_df.mean().to_dict()
    for metric, label_values in summary.items():
        for label, score in label_values.items():
            tt_writer.add_scalar(f"benchmark_{metric}/{label}", score)
    df.to_csv((args.save_folder / 'results.csv'), index=False)
        

if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
