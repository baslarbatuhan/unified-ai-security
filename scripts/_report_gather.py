"""One-off gathering script for the system report. Throwaway."""
import json, csv, yaml
from pathlib import Path
from collections import Counter

ROOT = Path("/home/batuhan/unified-ai-security")

print("=" * 70)
print("CONFIGURED TARGETS")
print("=" * 70)
with open(ROOT / "external_eval/targets.yaml") as f:
    d = yaml.safe_load(f)
for t in d["targets"]:
    print(f"  {t['id']:30s} type={t['type']:12s} enabled={t.get('enabled',True)} has_tools={t.get('has_tools',False)}")

print()
print("=" * 70)
print("TEST FILES BREAKDOWN (40 files)")
print("=" * 70)
test_files = sorted((ROOT / "tests").glob("test_*.py"))
for tf in test_files:
    name = tf.stem.replace("test_", "")
    print(f"  {name}")

print()
print("=" * 70)
print("CASE STUDY RUNS")
print("=" * 70)
case_studies = [
    ("run_20260523002424_2f190250c035", "Web target prompt_injection baseline"),
    ("run_20260523002553_f18e95bd163f", "API target prompt_injection (A/B partner)"),
    ("run_20260523015403_6d474e7f4f54", "API rag_poisoning (timeout-free baseline)"),
    ("run_20260523015641_347a75cbc9d1", "Web rag_poisoning (timeout-induced FN/FP)"),
    ("run_20260524180533_b4a862d3d46b", "agency_social isolated (prompt_guard off)"),
    ("run_20260524190023_747078c17118", "agency_social post-fix-A+C"),
]
for rid, desc in case_studies:
    rd = ROOT / "runs" / rid
    if not rd.exists():
        print(f"  [MISSING] {rid}")
        continue
    m = json.load(open(rd / "manifest.json"))
    rows = list(csv.DictReader(open(rd / "results.csv")))
    decisions = Counter(r.get("gateway_decision", "") for r in rows)
    matches = sum(1 for r in rows if r.get("expected_decision") == r.get("gateway_decision"))
    tools_x = sum(1 for r in rows if r.get("tool_executed") == "1")
    print(f"  {rid}")
    print(f"    {desc}")
    target_id = m["target_id"]
    suite = m["suite"]
    print(f"    target={target_id:25s} suite={suite:18s} n={len(rows)}")
    print(f"    decisions={dict(decisions)}")
    print(f"    accuracy={matches}/{len(rows)}  tool_executed={tools_x}/{len(rows)}")
    print()

print("=" * 70)
print("DATASETS / ATTACK CORPUS")
print("=" * 70)
ds_dir = ROOT / "datasets"
for f in ds_dir.rglob("*.json"):
    rel = f.relative_to(ds_dir)
    try:
        d = json.load(open(f))
        if isinstance(d, list):
            n = len(d)
        elif isinstance(d, dict):
            for k in ("scenarios", "items", "cases", "documents", "samples"):
                if k in d and isinstance(d[k], list):
                    n = len(d[k])
                    break
            else:
                n = "(dict, no list field)"
        else:
            n = "?"
        size_kb = f.stat().st_size // 1024
        print(f"  {str(rel):50s} {n} items  ({size_kb} KB)")
    except Exception as e:
        print(f"  {rel}  (error: {e})")

print()
print("=" * 70)
print("CONFIGS")
print("=" * 70)
for f in (ROOT / "configs").iterdir():
    if f.suffix in (".yaml", ".py"):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:40s} ({size_kb} KB)")
