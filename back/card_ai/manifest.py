from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

# 集中式生产模型清单定义
PRODUCTION_MODEL_MANIFEST: dict[str, Any] = {
    "model_id": "v21_rule_rollout_512",
    "iteration": 25,
    "relative_path": "runs_deep_cfr/deep_cfr_v21_fixed_rule_rollout_512/policy_iter_00025_baseline.pt",
    "expected_file_sha256": "0ffe95b878391a6f985c5a6fbc2c23581c8f9bf0396ae0e7044f3b2065c790a4",
    "expected_tensor_sha256": "5e6c929ef06e36c4a5dfd73feedc777aa21eeae5d1c977fa4219265ef56c11f1",
    "expected_file_size": 3460109,
    "hidden_dim": 512,
    "num_layers": 4,
    "feature_dim": 64,
    "action_space_size": 58,
    "prob_mode": "linear_norm",
    "objective_version": "rank_rollout_fixed_rule_v1",
}


def compute_file_sha256(file_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_tensor_sha256(file_path: str | Path) -> str:
    import torch

    ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("strategy_net", ckpt)
    th = hashlib.sha256()
    for k in sorted(sd.keys()):
        th.update(k.encode())
        th.update(sd[k].numpy().tobytes())
    return th.hexdigest()


def verify_production_model(base_dir: str | Path | None = None) -> tuple[bool, str, dict[str, Any]]:
    """严格校验生产模型完整性。
    
    返回 (is_valid, message, metadata_summary)
    """
    base = Path(base_dir or Path(__file__).resolve().parent.parent)
    model_path = base / PRODUCTION_MODEL_MANIFEST["relative_path"]

    meta = dict(PRODUCTION_MODEL_MANIFEST)
    meta["resolved_path"] = str(model_path.resolve())

    if not model_path.is_file():
        return False, f"生产模型文件不存在: {model_path}", meta

    # 1. 校验文件 64 位 SHA256
    file_sha = compute_file_sha256(model_path)
    meta["actual_file_sha256"] = file_sha
    meta["short_sha256"] = file_sha[:16]

    if file_sha != PRODUCTION_MODEL_MANIFEST["expected_file_sha256"]:
        return (
            False,
            f"文件哈希不匹配! 预期: {PRODUCTION_MODEL_MANIFEST['expected_file_sha256']}, 实际: {file_sha}",
            meta,
        )

    # 2. 校验文件大小
    actual_size = model_path.stat().st_size
    meta["actual_file_size"] = actual_size
    if actual_size != PRODUCTION_MODEL_MANIFEST["expected_file_size"]:
        return (
            False,
            f"文件大小不匹配! 预期: {PRODUCTION_MODEL_MANIFEST['expected_file_size']}, 实际: {actual_size}",
            meta,
        )

    # 3. 校验张量 SHA256
    try:
        tensor_sha = compute_tensor_sha256(model_path)
        meta["actual_tensor_sha256"] = tensor_sha
        if tensor_sha != PRODUCTION_MODEL_MANIFEST["expected_tensor_sha256"]:
            return (
                False,
                f"张量权重哈希不匹配! 预期: {PRODUCTION_MODEL_MANIFEST['expected_tensor_sha256']}, 实际: {tensor_sha}",
                meta,
            )
    except Exception as e:
        return False, f"张量权重解析异常: {e}", meta

    return True, "生产模型完整性校验 100% 通过", meta


if __name__ == "__main__":
    ok, msg, m = verify_production_model()
    print("Verification Result:", ok)
    print("Message:", msg)
    print("Metadata:", m)
