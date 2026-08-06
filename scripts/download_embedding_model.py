"""
Download Qwen3-Embedding-0.6B from HuggingFace.

Usage:
    python scripts/download_embedding_model.py
    python scripts/download_embedding_model.py --save-dir D:/models/Qwen3-Embedding-0.6B
"""

import argparse
import os
import sys


def download_with_snapshot(save_dir: str):
    """Use huggingface_hub to download (recommended)."""
    from huggingface_hub import snapshot_download

    print(f"Downloading Qwen/Qwen3-Embedding-0.6B to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    snapshot_download(
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Done.")


def download_with_transformers(save_dir: str):
    """Fallback: use transformers to load + save."""
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading Qwen/Qwen3-Embedding-0.6B via transformers...")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

    print(f"Saving to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--save-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "Qwen3-Embedding-0.6B"),
        help="Local directory to save the model",
    )
    args = parser.parse_args()

    try:
        download_with_snapshot(args.save_dir)
    except ImportError:
        print("huggingface_hub not available, falling back to transformers...")
        try:
            download_with_transformers(args.save_dir)
        except ImportError:
            print("Neither huggingface_hub nor transformers is installed.")
            print("Run: pip install huggingface_hub transformers")
            sys.exit(1)


if __name__ == "__main__":
    main()