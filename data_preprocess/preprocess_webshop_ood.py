"""
Preprocess WebShop OOD splits for reproducible training and evaluation.

Loads the full WebShop dataset (use_small=False, human_goals=True) which
gives ~12,087 crowd-sourced goals, classifies them into 7 categories,
splits by WebShop's original train/test+eval boundary, applies FPS
downsampling to the 'other' category, and writes a single JSON file
containing all split indices.

Usage:
  python -m examples.data_preprocess.preprocess_webshop_ood \
      --webshop_data env_data/webshop \
      --output env_data/webshop/webshop_ood_splits.json \
      --seed 0 \
      --other_train 1000 --other_val 100 \
      --embedding_model /path/to/Qwen3-Embedding-0.6B
"""

import argparse
import json
import logging
import os
import random
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WEBSHOP_ROOT = os.path.join(PROJECT_ROOT, 'agent_system/environments/env_package/webshop/webshop')


def main(args):
    # Ensure webshop modules can find data files
    os.environ["WEBSHOP_DATA"] = os.path.abspath(args.webshop_data)

    # Add paths for imports
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, WEBSHOP_ROOT)

    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals
    from agent_system.environments.env_package.webshop.envs import (
        classify_webshop_goal,
        farthest_point_sampling,
    )

    webshop_data = os.path.abspath(args.webshop_data)

    # --- Load products (human-goal subset for human_goals=True) ---
    file_path = os.path.join(webshop_data, 'items_shuffle_human.json')
    attr_path = os.path.join(webshop_data, 'items_ins_v2_human.json')
    logger.info(f"Loading products from {file_path}")
    t0 = time.time()
    all_products, product_item_dict, product_prices, _ = load_products(
        filepath=file_path, attrpath=attr_path, num_products=None, human_goals=True,
    )
    logger.info(f"Products loaded in {time.time()-t0:.1f}s, {len(all_products)} products")

    # --- Generate human goals ---
    # Seed *before* get_goals: get_human_goals uses random.sample for price_upper,
    # which affects instruction_text and downstream FPS embeddings.
    random.seed(args.seed)
    t1 = time.time()
    logger.info("Generating goals (human_goals=True)...")
    goals = get_goals(all_products, product_prices, human_goals=True)
    logger.info(f"Goals generated in {time.time()-t1:.1f}s, {len(goals)} goals")

    # Free memory
    del all_products, product_item_dict, product_prices
    import gc; gc.collect()

    # --- Shuffle with seed (matches runtime behavior) ---
    # Re-seed to decouple shuffle result from goal generation's random consumption.
    random.seed(args.seed)
    random.shuffle(goals)

    # --- Classify all goals ---
    logger.info("Classifying goals...")
    categories = {}
    goal_cats = []
    for i, g in enumerate(goals):
        cat = classify_webshop_goal(g.get('instruction_text', ''))
        goal_cats.append(cat)
        categories.setdefault(cat, []).append(i)

    logger.info("Category distribution (all goals):")
    for cat, idxs in sorted(categories.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {cat}: {len(idxs)}")

    # --- Split by WebShop convention ---
    # test: 0-499, eval: 500-1499, train: 1500+
    train_range = set(range(1500, len(goals)))
    val_range = set(range(0, 1500))

    id_categories = set(args.id_categories)
    ood_categories = set(args.ood_categories)

    # Partition into splits
    train_by_cat = {}
    val_by_cat = {}
    for cat, idxs in categories.items():
        train_by_cat[cat] = [i for i in idxs if i in train_range]
        val_by_cat[cat] = [i for i in idxs if i in val_range]

    logger.info("\nPer-split category counts:")
    logger.info(f"  {'Category':<15} {'Train(1500+)':>12} {'Val(0-1499)':>12}")
    for cat in sorted(categories.keys()):
        logger.info(f"  {cat:<15} {len(train_by_cat[cat]):>12} {len(val_by_cat[cat]):>12}")

    # --- FPS downsample 'other' ---
    def fps_downsample(idxs, n_target, split_name):
        if n_target is None or len(idxs) <= n_target:
            logger.info(f"  [{split_name}] other: {len(idxs)} goals, no downsampling needed")
            return idxs
        texts = [goals[i]['instruction_text'] for i in idxs]
        logger.info(f"  [{split_name}] FPS downsampling other: {len(idxs)} -> {n_target} "
                     f"(model={os.path.basename(args.embedding_model)})")
        t = time.time()
        selected_local = farthest_point_sampling(
            texts, n_target, args.embedding_model, seed=args.seed,
        )
        result = [idxs[j] for j in selected_local]
        logger.info(f"  [{split_name}] FPS done in {time.time()-t:.1f}s")
        return result

    other_train = fps_downsample(train_by_cat.get('other', []), args.other_train, 'train')
    other_val = fps_downsample(val_by_cat.get('other', []), args.other_val, 'val')

    # --- Assemble splits ---
    # Train: ID categories from train range, other already downsampled
    train_idxs = []
    train_counts = {}
    for cat in sorted(id_categories):
        if cat == 'other':
            cat_idxs = other_train
        else:
            cat_idxs = train_by_cat.get(cat, [])
        train_idxs.extend(cat_idxs)
        train_counts[cat] = len(cat_idxs)

    # Val ID: ID categories from val range, other already downsampled
    val_id_idxs = []
    val_id_counts = {}
    for cat in sorted(id_categories):
        if cat == 'other':
            cat_idxs = other_val
        else:
            cat_idxs = val_by_cat.get(cat, [])
        val_id_idxs.extend(cat_idxs)
        val_id_counts[cat] = len(cat_idxs)

    # Val OOD: OOD categories from val range
    val_ood_idxs = []
    val_ood_counts = {}
    for cat in sorted(ood_categories):
        cat_idxs = val_by_cat.get(cat, [])
        val_ood_idxs.extend(cat_idxs)
        val_ood_counts[cat] = len(cat_idxs)

    # --- Output ---
    output = {
        "meta": {
            "seed": args.seed,
            "total_goals": len(goals),
            "human_goals": True,
            "data_files": ["items_shuffle.json", "items_ins_v2.json"],
            "id_categories": sorted(id_categories),
            "ood_categories": sorted(ood_categories),
            "other_fps": {
                "train": args.other_train,
                "val": args.other_val,
                "embedding_model": os.path.basename(args.embedding_model),
            },
        },
        "train": {
            "goal_idxs": sorted(train_idxs),
            "count": len(train_idxs),
            "by_category": train_counts,
        },
        "val_id": {
            "goal_idxs": sorted(val_id_idxs),
            "count": len(val_id_idxs),
            "by_category": val_id_counts,
        },
        "val_ood": {
            "goal_idxs": sorted(val_ood_idxs),
            "count": len(val_ood_idxs),
            "by_category": val_ood_counts,
        },
    }

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nOutput written to: {output_path}")
    logger.info(f"  train: {output['train']['count']} goals {output['train']['by_category']}")
    logger.info(f"  val_id: {output['val_id']['count']} goals {output['val_id']['by_category']}")
    logger.info(f"  val_ood: {output['val_ood']['count']} goals {output['val_ood']['by_category']}")
    logger.info(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess WebShop OOD splits (human_goals=True, full data).",
    )
    parser.add_argument(
        "--webshop_data",
        default="env_data/webshop",
        help="Directory containing items_shuffle.json and items_ins_v2.json.",
    )
    parser.add_argument(
        "--output",
        default="env_data/webshop/webshop_ood_splits.json",
        help="Path to write the output splits JSON.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffle and FPS.")
    parser.add_argument("--other_train", type=int, default=1000,
                        help="Downsample 'other' in train split to this count via FPS.")
    parser.add_argument("--other_val", type=int, default=100,
                        help="Downsample 'other' in val split to this count via FPS.")
    parser.add_argument(
        "--embedding_model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Path to sentence-transformer model for FPS embeddings.",
    )
    parser.add_argument(
        "--id_categories",
        nargs="+",
        default=["apparel", "footwear", "electronics", "other"],
        help="In-distribution category names.",
    )
    parser.add_argument(
        "--ood_categories",
        nargs="+",
        default=["home_decor", "accessories", "beauty_health"],
        help="Out-of-distribution category names.",
    )
    args = parser.parse_args()
    main(args)
