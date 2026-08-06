"""Extract human subset from full items_shuffle.json."""
import json, os

SHUFFLE = r"data\webshop_data\items_shuffle.json"
HUMAN_INS = r"data\webshop_data\items_human_ins.json"
OUT = r"data\webshop_data\items_shuffle_human.json"

# Also filter items_ins_v2 if available
INS = r"data\webshop_data\items_ins_v2.json"
INS_OUT = r"data\webshop_data\items_ins_v2_human.json"

print("Loading human ASINs...")
with open(HUMAN_INS) as f:
    human_ins = json.load(f)
human_asins = set(human_ins.keys())
print(f"  {len(human_asins)} human-goal ASINs")

# ---- items_shuffle ----
print("Filtering items_shuffle.json -> items_shuffle_human.json...")
with open(SHUFFLE) as f:
    products = json.load(f)
print(f"  {len(products)} total products")
human_products = [p for p in products if p.get("asin") in human_asins]
with open(OUT, "w") as f:
    json.dump(human_products, f)
print(f"  Saved {len(human_products)} products ({os.path.getsize(OUT)/1024/1024:.1f} MB)")

# ---- items_ins_v2 ----
if os.path.exists(INS):
    print("Filtering items_ins_v2.json -> items_ins_v2_human.json...")
    with open(INS) as f:
        attrs = json.load(f)
    if isinstance(attrs, list):
        human_attrs = [a for a in attrs if a.get("asin") in human_asins]
    else:
        human_attrs = {k: v for k, v in attrs.items() if k in human_asins}
    with open(INS_OUT, "w") as f:
        json.dump(human_attrs, f)
    print(f"  Saved {len(human_attrs)} entries ({os.path.getsize(INS_OUT)/1024/1024:.1f} MB)")

print("Done!")