from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
from typing import Any

import hydra
import numpy as np
import onnxruntime as ort
import torch
from omegaconf import OmegaConf
from onnxruntime.quantization import QuantType, quantize_dynamic

from deepfakedet.data import BaseDataModule
from deepfakedet.models import ViTClassifier


def load_model_from_checkpoint(
    ckpt: dict[str, Any],
) -> tuple[ViTClassifier, Any]:
    cfg = OmegaConf.create(ckpt.get("cfg"))
    model: ViTClassifier = hydra.utils.instantiate(cfg.model)

    # extract model state from LightningModule
    model_state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(model_state)
    return model, cfg


def export_onnx(model: ViTClassifier, output_dir: Path, cfg: Any) -> Path:
    model.eval()
    model.cpu()

    datamodule: BaseDataModule = hydra.utils.instantiate(cfg.data)
    img_size: tuple[int, int] = datamodule.img_size
    num_channels: int = datamodule.num_channels
    input_shape = (1, num_channels, *img_size)

    input_tensor = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        (input_tensor,),
        f"{str(output_dir)}/model.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    return Path(output_dir) / "model.onnx"


def quantize_onnx(onnx_path: Path, output_dir: Path, quant_type: str) -> Path:
    quant_type_str = quant_type.lower()
    quant = getattr(QuantType, quant_type)

    output_path = output_dir / f"model_{quant_type_str}.onnx"
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=quant,
    )
    return output_path


def benchmark(onnx_path: Path, n_runs: int = 100) -> dict[str, float]:
    size_mb = Path(onnx_path).stat().st_size
    session = ort.InferenceSession(str(onnx_path))
    shape = session.get_inputs()[0].shape
    dummy = np.random.randn(1, *shape[1:]).astype(np.float32)

    times = []
    for _ in range(n_runs):
        start = perf_counter()
        session.run(None, {"input": dummy})
        times.append((perf_counter() - start) * 1000)  # convert to ms

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))

    return {
        "size_mb": size_mb / (1024**2),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str)
    parser.add_argument("--output-dir", "-o", type=str, default="exports/")
    parser.add_argument(
        "--quant-type",
        type=str,
        default="QInt8",
        help="QuantType member name (e.g. QInt8, QUInt8). Pass 'none' to skip.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model, cfg = load_model_from_checkpoint(ckpt)

    # merge LoRA with ViT model
    model.merge_adapter()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(model, output_dir, cfg)

    if args.quant_type != "none":
        quant = args.quant_type
        quantized_onnx_path = quantize_onnx(
            onnx_path, output_dir=output_dir, quant_type=quant
        )

    benchmark_onnx = benchmark(onnx_path, n_runs=100)
    print(benchmark_onnx)

    if args.quant_type != "none":
        benchmark_quantized_onnx = benchmark(quantized_onnx_path, n_runs=100)
        print(benchmark_quantized_onnx)
        size_reduction = (
            1 - benchmark_quantized_onnx["size_mb"] / benchmark_onnx["size_mb"]
        ) * 100
    latency_reduction = (
        1 - benchmark_quantized_onnx["mean_ms"] / benchmark_onnx["mean_ms"]
    ) * 100

    print(
        f"\n{'Metric':<20} {'FP32':>10} {'Quantized':>12} {'Delta':>10} \
            {'Improvement':>12}"
    )
    print("-" * 68)
    print(
        f"{'Size (MB)':<20} {benchmark_onnx['size_mb']:>10.1f} "
        f"{benchmark_quantized_onnx['size_mb']:>12.1f} "
        f"{benchmark_quantized_onnx['size_mb'] - benchmark_onnx['size_mb']:>+10.1f} "
        f"{size_reduction:>11.1f}%"
    )
    print(
        f"{'Latency mean (ms)':<20} {benchmark_onnx['mean_ms']:>10.2f} "
        f"{benchmark_quantized_onnx['mean_ms']:>12.2f} "
        f"{benchmark_quantized_onnx['mean_ms'] - benchmark_onnx['mean_ms']:>+10.2f} "
        f"{latency_reduction:>11.1f}%"
    )
    print(
        f"{'Latency std (ms)':<20} {benchmark_onnx['std_ms']:>10.2f} "
        f"{benchmark_quantized_onnx['std_ms']:>12.2f} "
        f"{benchmark_quantized_onnx['std_ms'] - benchmark_onnx['std_ms']:>+10.2f}"
    )
