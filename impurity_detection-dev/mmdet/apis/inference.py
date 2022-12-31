import warnings

import mmcv,cv2
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, device)
    else:
        return _inference_generator(model, imgs, img_transform, device)


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)


def json_result_2(result, resize_scale, score_thr=0.2):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    inds = np.where(bboxes[:, -1] > score_thr)[0]
    valid_boxes = bboxes[inds]
    num_boxes = len(valid_boxes)

    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)
    labels = labels[inds]

    per_image_result = []
    for ind in range(num_boxes):
        x_min, y_min, x_max, y_max, score = (valid_boxes[ind]).tolist()[:]
        per_det_result = {}
        dict_bbox={}
        dict_bbox["xmin"] = int(x_min*resize_scale[0])
        dict_bbox["ymin"] = int(y_min*resize_scale[1])
        dict_bbox["xmax"] = int(x_max*resize_scale[0])
        dict_bbox["ymax"] = int(y_max*resize_scale[1])
        per_det_result["bbox"] = dict_bbox
        per_det_result["mask"] = []
        per_det_result["score"] = score
        per_det_result["label"] = labels[ind].item()
        per_image_result.append(per_det_result)

    return per_image_result, num_boxes


def write_result(result, detla = 4096, l_use_scale=False, src_size=None):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    if l_use_scale:
        assert isinstance(src_size, (list, tuple)), "when using scale model, you must assign src_size, " \
                                                    "but got src_size {}".format(type(src_size))
        bboxes[:, 0] = bboxes[:, 0] * (src_size[1]/2048.0)
        bboxes[:, 1] = bboxes[:, 1] * (src_size[0]/1024.0)
        bboxes[:, 2] = bboxes[:, 2] * (src_size[1]/2048.0)
        bboxes[:, 3] = bboxes[:, 3] * (src_size[0]/1024.0)
    else:
        bboxes = bboxes*2
        bboxes[:, 0] = bboxes[:, 0] + detla
        bboxes[:, 2] = bboxes[:, 2] + detla
        bboxes[:, -1] = bboxes[:, -1]/2.0

    return bboxes

def save_result_1(img, bboxes, class_names, score_thr=0.3, out_file=None):
    # labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bboxes)]
    # labels = np.concatenate(labels)
    labels = np.zeros(len(bboxes), dtype=np.int8)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        bbox_color='red',
        text_color='red',
        thickness=6,
        font_scale=2.,
        show=out_file is None,
        out_file=out_file)

def json_result(result, src_size, resize_scale, score_thr=0.3):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    inds = np.where(bboxes[:, -1] > score_thr)[0]
    valid_boxes = bboxes[inds]
    num_boxes = len(valid_boxes)

    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)
    labels = labels[inds]

    valid_mask = []
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        for i in inds:
            per_mask = maskUtils.decode(segms[i]).astype(np.uint8)
            valid_mask.append(per_mask)

    valid_num = 0
    per_image_result = []
    for ind in range(num_boxes):
        x_min, y_min, x_max, y_max, score = (valid_boxes[ind]).tolist()[:]
        if abs(x_max - x_min)*abs(y_max - y_min)>20:#>min_area:
            valid_num += 1
            per_det_result = {}
            dict_bbox={}
            resize_xmin = int(x_min*resize_scale[0])
            resize_ymin = int(y_min*resize_scale[1])
            resize_xmax = int(x_max*resize_scale[0])
            resize_ymax = int(y_max*resize_scale[1])
            dict_bbox["xmin"] = resize_xmin
            dict_bbox["ymin"] = resize_ymin
            dict_bbox["xmax"] = resize_xmax
            dict_bbox["ymax"] = resize_ymax
            per_det_result["bbox"] = dict_bbox
            
            resize_mask = cv2.resize(valid_mask[ind], dsize=src_size).astype(np.uint8)
            resize_mask = resize_mask[:,:,None]
            contours, _ = cv2.findContours(resize_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_area = np.array([cv2.contourArea(contour) for contour in contours])
            contours_list = []
            sorted_idx = np.argsort(contours_area)[::-1]
            valid_contours = np.array(contours)[sorted_idx][:3]
            for idx, contour in enumerate(valid_contours):
                if cv2.contourArea(contour) > 25:
                    contours_list.append(np.squeeze(contour).tolist())
            per_det_result["mask"] = contours_list #[x, y]
            per_det_result["score"] = score
            per_det_result["label"] = labels[ind].item()
            per_image_result.append(per_det_result)

    return per_image_result, valid_num

def show_result_2(img, result, class_names, score_thr=0.1, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            """
            _, M = ignore_label_index.size()
            a, *b = ignore_label_index.chunk(M, dim=1)
            """
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(img.copy(), bboxes, labels, class_names=class_names,
                           score_thr=score_thr, bbox_color='red', text_color='red', thickness=4,
                           font_scale=1.1, show=out_file is None, out_file=out_file)


def show_result_1(img, result, class_names, score_thr=0.1, out_file=None, use_mask=False):
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    valid_inds = np.where(bboxes[:, -1] > score_thr)[0]
    valid_bboxes = bboxes[valid_inds]
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    valid_labels = np.concatenate(labels)[valid_inds]
    
    if segm_result is not None and use_mask:
        segms = mmcv.concat_list(segm_result)
        for i in valid_inds:
            # color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            # img[mask] = img[mask] * 0.5 + color_mask * 0.5

            resize_mask = mask.astype(np.uint8)[:,:,None]
            contours, _ = cv2.findContours(resize_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_area = np.array([cv2.contourArea(contour) for contour in contours])
            contours_list = []
            sorted_idx = np.argsort(contours_area)[::-1]
            valid_contours = np.array(contours)[sorted_idx][:5]
            for idx, contour in enumerate(valid_contours):
                if cv2.contourArea(contour) > 25:
                    contours_list.append(contour)
                    cv2.polylines(img, [contour], True, (0, 0, 255), 2)

    mmcv.imshow_det_bboxes(img.copy(), valid_bboxes, valid_labels,
                           class_names=class_names, score_thr=score_thr,
                           bbox_color='red', text_color='red', thickness=2,
                           font_scale=1.1, show=out_file is None, out_file=out_file)

def save_result(img, result, class_names, score_thr=0.3, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.8 + color_mask * 0.2
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file)
