import argparse
import importlib
import json
import os
import sys
from typing import Dict, Tuple

import torch
from torch.profiler import ProfilerActivity, profile


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Parameters and FLOPs from a .pth checkpoint"
    )
    parser.add_argument("--pth", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "ldonet_s", "ldonet_t", "sf2", "sf2_trans", "compnet", "co3", "cc"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--num-classes",
        "--num-class",
        dest="num_classes",
        type=int,
        default=193,
        help="Override class count",
    )
    parser.add_argument("--weight", type=float, default=0.7, help="Fusion weight used by constructor")
    parser.add_argument("--input-h", type=int, default=128, help="Input image height")
    parser.add_argument("--input-w", type=int, default=128, help="Input image width")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for FLOPs")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser.parse_args()


def _extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "student", "teacher"]:
            value = ckpt_obj.get(key, None)
            if isinstance(value, dict) and value:
                if all(isinstance(v, torch.Tensor) for v in value.values()):
                    return value

        if ckpt_obj and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj

    raise ValueError("Unsupported checkpoint format: cannot find state_dict")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def infer_model_type(state_dict: Dict[str, torch.Tensor]) -> str:
    # CCNet (models/CCnet.py)
    if "arclayer_.weight" in state_dict:
        return "cc"

    # CompNet (models/Compnet.py)
    comp_fc = state_dict.get("fc.weight", None)
    comp_arc = state_dict.get("arclayer.weight", None)
    if comp_fc is not None and comp_fc.ndim == 2 and comp_fc.shape[1] == 9708:
        return "compnet"
    if comp_arc is not None and comp_arc.ndim == 2 and comp_arc.shape[1] == 512:
        return "compnet"

    # CO3Net (models/CO3net.py)
    co3_arc = state_dict.get("arclayer.weight", None)
    if "fc1.weight" in state_dict or "cb1.gabor_conv2d2.gamma" in state_dict:
        return "co3"
    if co3_arc is not None and co3_arc.ndim == 2 and co3_arc.shape[1] == 2048:
        return "co3"

    # Legacy SF2Net (models/sf2net.py)
    if "fully_connection_for_vit_1.weight" in state_dict or "vit_0.cls_token" in state_dict:
        return "sf2"

    # New TranSF2 variant (models/TranSF2.py)
    if "fully_connection_for_vit.weight" in state_dict:
        return "sf2_trans"

    fc1_w = state_dict.get("fully_connection_1.weight", None)
    if fc1_w is not None and fc1_w.ndim == 2 and fc1_w.shape[1] == 864:
        return "ldonet_s"

    if "fully_connection_for_deg.weight" in state_dict:
        return "ldonet_t"

    raise ValueError("Cannot infer model type. Please pass --model-type manually.")


def infer_num_classes(state_dict: Dict[str, torch.Tensor], num_classes_override: int = None) -> int:
    if num_classes_override is not None:
        return int(num_classes_override)

    arc_w = state_dict.get("arcface.weight", None)
    if arc_w is None:
        arc_w = state_dict.get("arclayer.weight", None)
    if arc_w is None:
        arc_w = state_dict.get("arclayer_.weight", None)

    if arc_w is None:
        raise ValueError("Cannot infer num_classes from checkpoint. Please pass --num-classes.")

    return int(arc_w.shape[0])


def build_model(model_type: str, num_classes: int, weight: float) -> torch.nn.Module:
    if model_type == "cc":
        from models.CCnet import ccnet

        return ccnet(num_classes=num_classes, weight=weight)

    if model_type == "compnet":
        from models.Compnet import compnet

        return compnet(num_classes=num_classes)

    if model_type == "co3":
        from models.CO3net import co3net

        return co3net(num_classes=num_classes)

    if model_type == "ldonet_s":
        from models.LDONet_S import LDONet_S

        return LDONet_S(label_num=num_classes, weight=weight)

    if model_type == "ldonet_t":
        from models.LDONet_T import LDONet_T

        return LDONet_T(label_num=num_classes, weight=weight)

    if model_type == "sf2":
        from models.sf2net import SF2Net

        return SF2Net(label_num=num_classes, weight=weight)

    if model_type == "sf2_trans":
        try:
            sf2_trans_module = importlib.import_module("models.TranSF2")
        except ModuleNotFoundError as exc:
            raise ValueError(
                "Model type 'sf2_trans' requires models/TranSF2.py, but it is not present in this workspace."
            ) from exc

        SF2Net = getattr(sf2_trans_module, "TranSF2Net")

        return SF2Net(label_num=num_classes, weight=weight)

    raise ValueError(f"Unsupported model_type: {model_type}")


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops(model: torch.nn.Module, batch_size: int, input_h: int, input_w: int) -> int:
    dummy = torch.randn(batch_size, 1, input_h, input_w, dtype=torch.float32)
    model.eval()

    with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=False) as prof:
        with torch.no_grad():
            model(dummy, None)

    total_flops = 0
    for event in prof.key_averages():
        event_flops = getattr(event, "flops", 0)
        if event_flops is not None:
            total_flops += int(event_flops)
    return total_flops


def human_readable_count(value: int) -> str:
    units = ["", "K", "M", "B", "T"]
    v = float(value)
    idx = 0
    while v >= 1000.0 and idx < len(units) - 1:
        v /= 1000.0
        idx += 1
    return f"{v:.3f}{units[idx]}"


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.pth):
        raise FileNotFoundError(f"Checkpoint not found: {args.pth}")

    ckpt_obj = torch.load(args.pth, map_location="cpu")
    state_dict = _extract_state_dict(ckpt_obj)
    state_dict = _strip_module_prefix(state_dict)

    model_type = infer_model_type(state_dict) if args.model_type == "auto" else args.model_type
    num_classes = infer_num_classes(state_dict, args.num_classes)

    model = build_model(
        model_type=model_type,
        num_classes=num_classes,
        weight=args.weight,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    params_total, params_trainable = count_parameters(model)
    flops = compute_flops(model, args.batch_size, args.input_h, args.input_w)

    result = {
        "checkpoint": os.path.abspath(args.pth),
        "model_type": model_type,
        "num_classes": num_classes,
        "input_shape": [args.batch_size, 1, args.input_h, args.input_w],
        "params_total": int(params_total),
        "params_total_human": human_readable_count(params_total),
        "params_trainable": int(params_trainable),
        "params_trainable_human": human_readable_count(params_trainable),
        "flops": int(flops),
        "flops_human": human_readable_count(flops),
        "gflops": float(flops) / 1e9,
        "state_dict_missing_keys": list(missing),
        "state_dict_unexpected_keys": list(unexpected),
    }

    mismatched = len(missing) + len(unexpected)
    if mismatched > 0:
        result["load_status"] = "partial"
    else:
        result["load_status"] = "clean"

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("=" * 72)
    print("Model Complexity Report")
    print("=" * 72)
    print(f"Checkpoint           : {result['checkpoint']}")
    print(f"Model Type           : {result['model_type']}")
    print(f"Num Classes          : {result['num_classes']}")
    print(f"Input Shape          : {result['input_shape']}")
    print(f"Load Status          : {result['load_status']}")
    print("-" * 72)
    print(f"Parameters (Total)   : {result['params_total']} ({result['params_total_human']})")
    print(f"Parameters (Trainable): {result['params_trainable']} ({result['params_trainable_human']})")
    print(f"FLOPs                : {result['flops']} ({result['flops_human']}, {result['gflops']:.4f} GFLOPs)")
    print("-" * 72)
    if result["state_dict_missing_keys"]:
        print(f"Missing keys ({len(result['state_dict_missing_keys'])}):")
        for k in result["state_dict_missing_keys"]:
            print(f"  - {k}")
    if result["state_dict_unexpected_keys"]:
        print(f"Unexpected keys ({len(result['state_dict_unexpected_keys'])}):")
        for k in result["state_dict_unexpected_keys"]:
            print(f"  - {k}")
    if not result["state_dict_missing_keys"] and not result["state_dict_unexpected_keys"]:
        print("State dict loaded cleanly.")
    elif len(result["state_dict_missing_keys"]) + len(result["state_dict_unexpected_keys"]) > 20:
        print("Hint: large key mismatch usually means wrong --model-type or wrong architecture branch.")
    print("=" * 72)


if __name__ == "__main__":
    main()
