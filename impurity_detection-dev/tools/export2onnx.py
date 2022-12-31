
import argparse

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default="/root/project/pytorch/highspeed.v3/configs/hrnet/cascade_rcnn_hrnetv2p_w32_v2.py",
                        help='train config file path')
    parser.add_argument('--out', default="/root/project/pytorch/highspeed.v3/ckpt_dirs/impurity.onnx",
                        help='output ONNX file')
    parser.add_argument('--checkpoint',
                        default="/root/project/pytorch/highspeed.v3/ckpt_dirs/cascade_rcnn_hrnetv2p_w32_impurity_v2.bak/epoch_40.pth",
                        help='checkpoint file of the model')
    parser.add_argument(
        '--shape', type=int, nargs='+', default=(2048, 1024), help='input image size')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if len(args.shape) == 1:
        img_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        img_shape = (1, 3) + tuple(args.shape)
    elif len(args.shape) == 4:
        img_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    dummy_input = torch.randn(*img_shape, device='cuda')

    cfg = Config.fromfile(args.config)
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    if args.checkpoint:
        _ = load_checkpoint(model, args.checkpoint)

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'ONNX exporting is currently not currently supported with {}'.
            format(model.__class__.__name__))

    torch.onnx.export(model, dummy_input, args.out, verbose=True)


if __name__ == '__main__':
    main()