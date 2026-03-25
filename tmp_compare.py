import json

pairs = [("A", "tuning_A"), ("B", "tuning_B"), ("C", "tuning_C"), ("D", "tuning_D")]
print(f"{'Name':<6} {'R@10':>8} {'delta R':>8} {'MRR':>8} {'delta MRR':>10}")
print("-" * 46)
for name, fname in pairs:
    try:
        d = json.load(open(f"data/m2/{fname}.json"))
        m = d["metrics"]
        bm = m["bm25"]
        gf = m["graph_full"]
        dr = gf["recall_at_10"] - bm["recall_at_10"]
        dm = gf["mrr"] - bm["mrr"]
        print(f"{name:<6} {gf['recall_at_10']:>8.4f} {dr:>+8.4f} {gf['mrr']:>8.4f} {dm:>+10.4f}")
    except Exception as e:
        print(f"{name:<6} not ready ({e})")
