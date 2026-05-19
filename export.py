from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
from typing import Any

import hydra
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from omegaconf import OmegaConf
from onnxruntime.quantization import QuantType, quantize_dynamic

from deepfakedet.data import BaseDataModule
from deepfakedet.models import ViTClassifier


def get_cfg_from_checkpoint(ckpt: dict[str, Any]) -> Any:
    return OmegaConf.create(ckpt.get("cfg"))


def load_model_from_checkpoint(
    ckpt: dict[str, Any],
) -> tuple[ViTClassifier, Any]:
    cfg = get_cfg_from_checkpoint(ckpt)
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


def quantize_onnx(
    onnx_path: Path, output_dir: Path, quant_type: str, source_path: Path | None = None
) -> Path:
    quant_type_str = quant_type.lower()
    quant = getattr(QuantType, quant_type)

    input_path = source_path if source_path is not None else onnx_path
    output_path = output_dir / f"model_{quant_type_str}.onnx"
    quantize_dynamic(
        model_input=str(input_path),
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
        "--quant-types",
        nargs="+",
        help="One or more QuantType names (e.g. QInt8 QUInt8 QInt4) or 'none' to skip "
        "quantization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing model.onnx if present.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    available = {p.stem: p for p in output_dir.glob("*.onnx")}

    onnx_path = output_dir / "model.onnx"

    if "model" in available and not args.overwrite:
        print(f"Found existing {onnx_path}, skipping export. Use --overwrite to force.")
        cfg = get_cfg_from_checkpoint(ckpt)
    else:
        model, cfg = load_model_from_checkpoint(ckpt)
        model.merge_adapter()
        onnx_path = export_onnx(model, output_dir, cfg)

    model_paths: dict[str, Path] = {"none": onnx_path}

    # pre-populate from already-exported quantized models
    for quant_type in args.quant_types or []:
        if quant_type == "none":
            continue
        stem = f"model_{quant_type.lower()}"
        if stem in available and not args.overwrite:
            print(
                f"Found existing {available[stem]}, skipping. Use --overwrite to force."
            )
            model_paths[quant_type] = available[stem]

    # generate any that are still missing
    for quant_type in args.quant_types or []:
        if quant_type == "none" or quant_type in model_paths:
            continue

        if quant_type in ("QInt4", "QUInt4"):
            int8_source = model_paths.get("QInt8") or model_paths.get("QUInt8")
            if int8_source is None:
                print(
                    f"Skipping {quant_type} — requires QInt8 or QUInt8 to be generated"
                    " first."
                )
                continue
            quantized_path = quantize_onnx(
                onnx_path=onnx_path,
                output_dir=output_dir,
                quant_type=quant_type,
                source_path=int8_source,
            )
        else:
            quantized_path = quantize_onnx(
                onnx_path=onnx_path, output_dir=output_dir, quant_type=quant_type
            )

        model_paths[quant_type] = quantized_path

    # benchmark only base + requested types (in arg order)
    to_benchmark = ["none"] + [
        qt for qt in (args.quant_types or []) if qt in model_paths and qt != "none"
    ]
    benchmark_results = []

    for model_name in to_benchmark:
        benchmark_result = benchmark(model_paths[model_name], n_runs=100)
        benchmark_result["name"] = model_name
        benchmark_results.append(benchmark_result)

    df = pd.DataFrame(benchmark_results)

    base = df[df["name"] == "none"].iloc[0]
    df["size_delta_%"] = (
        (df["size_mb"] - base["size_mb"]) / base["size_mb"] * 100
    ).round(2)
    df["latency_delta_%"] = (
        (df["mean_ms"] - base["mean_ms"]) / base["mean_ms"] * 100
    ).round(2)

    print(df.to_string(index=False))
