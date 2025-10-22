"""
under /workspace
python3 ViTALiTy/src/quantize_model.py \
  --model-path models/vitality_train/best_checkpoint.pth \
  --model deit_tiny_patch16_224 \
  --degree 1 \
  --data-path /path/to/imagenet/val \
  --out models/vitality_train/ \
  --num-calib 512 \
  --workers 8

"""

import os
import argparse
from pathlib import Path
from timm import create_model
import torch
import torch.nn as nn
import models

from datasets import build_dataset
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import onnxruntime as ort


def get_args_parser():
    parser = argparse.ArgumentParser('Quantization of float vitality models', add_help=False)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--model", default="deit_tiny_patch16_224", type=str)
    parser.add_argument("--degree", default=1, type=int, choices=[1,2])
    parser.add_argument("--bits-act", default=8, type=int)
    parser.add_argument("--bits-wt", default=8, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--img-size", default=224, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--out", default="ptq_out", type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--data-path', default='/srv/datasets/imagenet/', type=str, help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument("--export-onnx", type=bool, default=True)
    parser.add_argument("--num-calib", type=int, default=512)
    parser.add_argument("--pin-mem", action="store_true", default=False)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    return parser


class DataReader(CalibrationDataReader):
    def __init__(self, dataloader, max_samples: int, input_name: str = "input"):
        self.dataloader = dataloader
        self.max_samples = max_samples
        self.input_name = input_name
        self._it = None
        self._seen = 0
        self.rewind()

    def get_next(self):
        if self._seen >= self.max_samples:
            return None
        try:
            batch, _ = next(self._it) 
        except StopIteration:
            return None
        self._seen += batch.shape[0]
        return {self.input_name: batch.cpu().numpy()}  

    def rewind(self):
        self._it = iter(self.dataloader)
        self._seen = 0


class _ExportLogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def quantize_model(model: nn.Module, args: argparse.Namespace, val_loader) -> str:
    # Export the float model to ONNX
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=args.device)
    onnx_path = os.path.join(args.out, "model_float.onnx")
    export_model = _ExportLogitsOnly(model).eval().to(args.device)
    torch.onnx.export(
        export_model, 
        dummy, 
        onnx_path,
        input_names=["input"], 
        output_names=["logits"],
        opset_version=17, 
        do_constant_folding=True,
        dynamic_axes={
            "input":  {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    num_calib = args.num_calib
    calib_subset  = torch.utils.data.Subset(val_loader, list(range(num_calib)))
    calib_loader  = torch.utils.data.DataLoader(
        calib_subset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False
    )
    calib_data = DataReader(calib_loader, max_samples=args.num_calib, input_name="input")
    quant_onnx_path = os.path.join(args.out, "model_quantized.onnx")
    quantize_static(
        model_input=onnx_path,
        model_output=quant_onnx_path,
        calibration_data_reader=calib_data,
        per_channel=True,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul","Gemm"],
        extra_options={"ActivationSymmetric": False},
        quant_format=QuantFormat.QOperator
    )
    return quant_onnx_path


def evaluate_onnx(onnx_path: str,
                  dataset_val,
                  args: argparse.Namespace) -> None:
    eval_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in eval_loader:
            x = imgs.cpu().numpy()
            y = labels.numpy()
            logits = sess.run([out_name], {in_name: x})[0] 
            pred = logits.argmax(axis=1)
            correct += (pred == y).sum()
            total   += y.shape[0]
    top1 = 100.0 * correct / max(1, total)
    print(f"[ACC] Quantized ONNX top-1 on {total} imgs: {top1:.2f}%")


def main(args):
    print(args)
    args.device = torch.device(args.device)

    args.nb_classes = 1000
    dataset_val, _ = build_dataset(is_train=False, args=args)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        vitality=True,
        degree=args.degree,
    )
    checkpoint = torch.load(args.model_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(args.device)
    model.eval()

    quant_model_path = quantize_model(model, args, dataset_val)
    evaluate_onnx(quant_model_path, dataset_val, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Quantization of float vitality models', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    main(args)
