import argparse
import json
import os
import time
import warnings
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis import collect_results_cpu, collect_results_gpu
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from pa_lib.evaluate_helper import (
    compute_ious,
    generate_pa_proposals,
)
from pa_lib.oln_ranker import generate_pa_proposals_with_oln
from pycocotools import mask as maskUtils


def proposals2json(masks, gt_ious, scores, img_id, orig_shape):
    masks = np.transpose(masks[:, :, :], (1, 2, 0))
    rles = maskUtils.encode(np.asfortranarray(masks))
    areas = maskUtils.area(rles)
    bboxes = maskUtils.toBbox(rles)
    anns = []
    for i, rle in enumerate(rles):
        new_rle = rle
        if type(new_rle["counts"]) == bytes:
            new_rle["counts"] = new_rle["counts"].decode("ascii")
        ann = {
            "segmentation": new_rle,
            "area": int(areas[i]),
            "bbox": [int(coord) for coord in bboxes[i]],
            "image_id": img_id,
            "category_id": 1,
            "rank": i,
            "gt_iou": gt_ious[i],
            "score": scores[i],
            "orig_shape": orig_shape,
        }
        anns.append(ann)
    return anns


def replace_filename_by_id(results, dataset):
    coco = dataset.coco
    filename_id_map = {}
    for i in dataset.img_ids:
        info = coco.load_imgs([i])[0]
        filename_id_map[info["file_name"]] = i
    for img_result in results:
        for result in img_result:
            result["image_id"] = filename_id_map[result["image_id"]]
    return results


# override single_gpu_test for PA
def single_gpu_test(
    model,
    data_loader,
    show=False,
    out_dir=None,
    topk=20,
    pa_func=generate_pa_proposals,
    use_gt_masks=True,
):
    """
    Function to extract masks generated by PA
        topk: extract topk masks per image at most
        pa_func: function to turn PA into masks
        use_gt_masks: whether the json comes with gt masks
            If so (e.g. COCO), we check the maximum IoU a pseudo mask has with
                GT; later we may filter out pseudo masks with high IoU
            If not (e.g. ImageNet), we skip the step above
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            _, pa_preds, gt_masks = model(return_loss=False, **data)
        if use_gt_masks:
            gt_masks = [inst_mask.to_ndarray() for inst_mask in gt_masks]
        for i, pa_pred in enumerate(pa_preds):
            pa_masks, scores = pa_func(pa_pred, **data)
            pa_masks = pa_masks[:topk]
            scores = scores[:topk]
            if use_gt_masks:
                ious = compute_ious(pa_masks, gt_masks[i]).max(axis=0)
            else:
                ious = np.zeros(len(pa_masks))
            pa_masks = np.stack(pa_masks)[:, :, :]
            img_meta = data["img_metas"].data[0][0]
            result = proposals2json(
                pa_masks, ious, scores, img_meta["ori_filename"], img_meta["ori_shape"]
            )
            results.append(result)
        batch_size = len(pa_preds)
        for _ in range(batch_size):
            prog_bar.update()
    return results


# override multi_gpu_test for PA
def multi_gpu_test(
    model,
    data_loader,
    tmpdir=None,
    gpu_collect=False,
    topk=20,
    pa_func=generate_pa_proposals,
    use_gt_masks=True,
):
    """
    See single_gpu_test
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            _, pa_preds, gt_masks = model(return_loss=False, **data)
            # encode mask results
            if use_gt_masks:
                gt_masks = [inst_mask.to_ndarray() for inst_mask in gt_masks]
            for i, pa_pred in enumerate(pa_preds):
                pa_masks, scores = pa_func(pa_pred, **data)
                if pa_masks is None:
                    continue
                pa_masks = pa_masks[:topk]
                scores = scores[:topk]
                if use_gt_masks:
                    if len(gt_masks[i]) == 0:
                        ious = np.zeros(len(pa_masks))
                    else:
                        ious = compute_ious(pa_masks, gt_masks[i]).max(axis=0)
                else:
                    ious = np.zeros(len(pa_masks))
                pa_masks = np.stack(pa_masks)[:, :, :]
                img_meta = data["img_metas"].data[0][0]
                result = proposals2json(
                    pa_masks,
                    ious,
                    scores,
                    img_meta["ori_filename"],
                    img_meta["ori_shape"],
                )
                results.append(result)

        if rank == 0:
            batch_size = len(pa_pred)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--eval-proposals",
        action="store_true",
        help="Whether to generate and eval proposals of PA",
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )
    parser.add_argument(
        "--show-score-thr",
        type=float,
        default=0.3,
        help="score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--test-ann-file", type=str, default=None)

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both "
            "specified, --options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if args.test_ann_file is not None:
        cfg.data.test.ann_file = args.test_ann_file
        print(f"overriding ann file with {args.test_ann_file}")
    cfg.model.pretrained = None
    if cfg.model.get("neck"):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get("rfp_backbone"):
            if cfg.model.neck.rfp_backbone.get("pretrained"):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    cfg.data.test.test_mode = False
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # # old versions did not save class info in checkpoints, this walkaround is
    # # for backward compatibility
    # if 'CLASSES' in checkpoint['meta']:
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES

    pa_mask_param = cfg.get("pa_to_masks", None)
    if pa_mask_param is not None:
        if pa_mask_param.oln_ranker_path is None:
            pa_func = partial(
                generate_pa_proposals,
                edge_thresh=pa_mask_param.edge_thresh,
                join_thresh=pa_mask_param.join_thresh,
                rank_method=pa_mask_param.rank_method,
                nms=pa_mask_param.nms,
                use_orientation=pa_mask_param.use_orientation,
                use_globalization=pa_mask_param.use_globalization,
                use_new_affinity=pa_mask_param.use_new_affinity,
                filter_by_edge_method=pa_mask_param.filter_by_edge_method,
                filter_by_edge_thresh=pa_mask_param.filter_by_edge_thresh,
                min_component_size=pa_mask_param.min_component_size,
            )
        else:
            # TODO: Hard-coded for now
            cfg_file = "/checkpoint/weiyaowang/dev/uvo/oln/configs/oln_box/oln_box.py"
            checkpoint = pa_mask_param.oln_ranker_path
            cfg = Config.fromfile(cfg_file)
            model_config = cfg.model
            oln_model = build_detector(model_config)
            _ = load_checkpoint(oln_model, checkpoint, map_location="cpu")
            oln_model.roi_head.test_cfg = None
            oln_model.cuda(torch.cuda.current_device())
            oln_model.eval()
            oln_model.rpn_head.objectness_assigner.pos_iou_thr = 1.0
            oln_model.rpn_head.objectness_assigner.min_pos_iou = 0.0
            pa_func = partial(
                generate_pa_proposals_with_oln,
                edge_thresh=pa_mask_param.edge_thresh,
                join_thresh=pa_mask_param.join_thresh,
                rank_method=pa_mask_param.rank_method,
                nms=pa_mask_param.nms,
                use_orientation=pa_mask_param.use_orientation,
                use_globalization=pa_mask_param.use_globalization,
                use_new_affinity=pa_mask_param.use_new_affinity,
                filter_by_edge_method=pa_mask_param.filter_by_edge_method,
                filter_by_edge_thresh=pa_mask_param.filter_by_edge_thresh,
                min_component_size=pa_mask_param.min_component_size,
                oln_model=oln_model,
            )
    else:
        pa_func = generate_pa_proposals

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model, data_loader, args.show, args.show_dir, pa_func=pa_func
        )
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect, pa_func=pa_func
        )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            out_dir = os.path.dirname(args.out)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            outputs = replace_filename_by_id(outputs, dataset)
            print(f"\nwriting results to {args.out}")
            json.dump(outputs, open(args.out, "w"))


if __name__ == "__main__":
    main()
