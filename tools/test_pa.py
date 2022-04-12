# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


import argparse
import os
import time
import warnings
from functools import partial

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis import collect_results_cpu, collect_results_gpu
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from pa_lib.evaluate_helper import (
    accumulate_result,
    compute_recall_normal,
    generate_pa_proposals,
)
from pa_lib.oln_ranker import generate_pa_proposals_with_oln


# specify the config file path for OLN model (when using OLN for ranking masks)
OLN_CFG_FILE = "../configs/oln_box/oln_box.py"


# override single_gpu_test for PA
def single_gpu_test(
    model,
    data_loader,
    show=False,
    out_dir=None,
    eval_proposals=False,
    generate_pa_func=generate_pa_proposals,
):
    """
    Function to evaluate PA (either compared with GT PA or evaluate generated masks)
        eval_proposals: if we construct masks and evaluate the masks
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
            pa_quality, pa_preds, gt_masks = model(return_loss=False, **data)
        if eval_proposals:
            gt_masks = [inst_mask.to_ndarray() for inst_mask in gt_masks]
            for i, gt_mask in enumerate(gt_masks):
                pa_masks, _ = generate_pa_func(
                    pa_preds[i],
                    **data,
                )
                recall = compute_recall_normal(gt_mask, pa_masks)
                # print(recall)
                pa_quality[i]["recall"] = recall
        batch_size = len(pa_quality)
        results.extend(pa_quality)
        for _ in range(batch_size):
            prog_bar.update()
    return results


# override multi_gpu_test for PA
def multi_gpu_test(
    model,
    data_loader,
    tmpdir=None,
    gpu_collect=False,
    eval_proposals=False,
    generate_pa_func=generate_pa_proposals,
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
            pa_quality, pa_preds, gt_masks = model(return_loss=False, **data)
            # encode mask results
            if eval_proposals:
                gt_masks = [inst_mask.to_ndarray() for inst_mask in gt_masks]
                for i, gt_mask in enumerate(gt_masks):
                    pa_masks, _ = generate_pa_func(
                        pa_preds[i],
                        **data,
                    )
                    recall = compute_recall_normal(gt_mask, pa_masks)
                    # print(recall)
                    pa_quality[i]["recall"] = recall
        results.extend(pa_quality)

        if rank == 0:
            batch_size = len(pa_quality)
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
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument(
        "--test-partition",
        type=str,
        default=None,
        help="override the test partition to avoid changing config file",
    )

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

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

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

    if args.test_partition is not None:
        print(f"overriding test partition {args.test_partition}")
        cfg.data.test.train_class = args.test_partition

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
    print(f"loading checkpoint {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    pa_mask_param = cfg.get("pa_to_masks", None)
    if pa_mask_param is not None:
        if pa_mask_param.oln_ranker_path is None:
            generate_pa_func = partial(
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
            # Load OLN model
            cfg_file = OLN_CFG_FILE
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

            generate_pa_func = partial(
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
        generate_pa_func = generate_pa_proposals

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            args.eval_proposals,
            generate_pa_func,
        )
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            args.eval_proposals,
            generate_pa_func,
        )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        if args.eval:
            # evaluate PA results
            metrics = accumulate_result(outputs)
            for eval_metric, total_perf in metrics.items():
                # last one in each perf means all proposals
                print(f"{eval_metric}: {total_perf}")


if __name__ == "__main__":
    main()
