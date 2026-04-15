from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_DIR = ROOT_DIR / "fintune" / "models"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 ModelScope 下载 Qwen2.5-72B-Instruct 模型")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="ModelScope 模型 ID，默认 Qwen/Qwen2.5-72B-Instruct",
    )
    parser.add_argument(
        "--target-dir",
        default=str(DEFAULT_TARGET_DIR),
        help="模型保存目录，默认保存到 X:/Extract_QLoRA_LLM_finetune/fintune/models",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="可选，ModelScope 访问令牌；也可通过环境变量 MODELSCOPE_API_TOKEN 提供",
    )
    parser.add_argument(
        "--revision",
        default="master",
        help="模型版本，默认 master",
    )
    parser.add_argument(
        "--local-name",
        default=None,
        help="本地子目录名，默认使用模型 ID 最后一段",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "未安装 modelscope，请先执行: pip install modelscope"
        ) from exc

    target_dir = Path(args.target_dir).resolve()
    ensure_dir(target_dir)

    local_name = args.local_name or args.model_id.rstrip("/").split("/")[-1]
    local_dir = target_dir / local_name
    ensure_dir(local_dir)

    token = args.token or os.getenv("MODELSCOPE_API_TOKEN")

    print(f"开始下载模型: {args.model_id}")
    print(f"保存目录: {local_dir}")
    if token:
        print("已检测到访问令牌，将使用认证下载")

    downloaded_path = snapshot_download(
        model_id=args.model_id,
        cache_dir=str(local_dir),
        local_dir=str(local_dir),
        local_files_only=False,
        revision=args.revision,
        user_agent={"source": "financial-report-finetune"},
        token=token,
    )

    print("下载完成")
    print(f"模型路径: {downloaded_path}")


if __name__ == "__main__":
    main()
