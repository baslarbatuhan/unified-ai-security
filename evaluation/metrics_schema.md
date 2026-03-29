# Metrics Schema

Standardized metrics format for all security modules.

## Purpose

All security modules (prompt_guard, rag_guard, output_agency_defense) must produce evaluation results in a common CSV format. This enables cross-module comparison, unified dashboards, and consistent reporting.

## CSV Format

All metrics CSV files in `runs/` must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| module | string | Module identifier: `prompt_guard`, `rag_guard`, `output_agency` |
| test_case | string | Test case identifier or prompt/query text (truncated to 80 chars) |
| decision | string | Module decision: `allow`, `sanitize`, `flag`, `block` |
| risk_score | float | Risk score between 0.0 and 1.0 |
| latency | int | Processing time in milliseconds |

## Extended Columns (optional, per-module)

Modules may add additional columns after the required five:

### prompt_guard
| Column | Type | Description |
|--------|------|-------------|
| actual_label | int | Ground truth: 0 = benign, 1 = attack |
| predicted_label | int | Model prediction: 0 = benign, 1 = attack |
| semantic_score | float | Cosine similarity to nearest known attack |
| threshold | float | Active threshold value |

### rag_guard
| Column | Type | Description |
|--------|------|-------------|
| query | string | Retrieval query |
| poisoned_count | int | Number of poisoned docs in top-k |
| total_retrieved | int | Total docs retrieved (k) |
| poison_ratio | float | poisoned_count / total_retrieved |

### output_agency
| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Session user identifier |
| tool | string | Tool name called |
| resource_id | string | Target resource ID (if applicable) |
| block_reason | string | Why blocked: `idor`, `unregistered_tool`, `enumeration`, `role_denied` |

## Output File Naming

```
runs/
├── prompt_metrics.csv          # Prompt guard evaluation results
├── rag_metrics.csv             # RAG guard evaluation results
├── agency_metrics.csv          # Agency guard evaluation results
├── week2_prompt_metrics.csv    # Threshold optimization sweep
└── baseline_vulnerability_report.json  # RAG baseline ASR report
```

## Aggregated Metrics

Each metrics CSV should support computation of these standard metrics:

| Metric | Formula |
|--------|---------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | 2 * P * R / (P + R) |
| FPR | FP / (FP + TN) |
| Accuracy | (TP + TN) / Total |
| ASR | Attacks that bypassed detection / Total attacks |

## Decision Thresholds

From `configs/secure_balanced.yaml`:

| Decision | Risk Score Range |
|----------|-----------------|
| allow | < 0.30 |
| sanitize | 0.30 - 0.60 |
| flag | 0.60 - 0.85 |
| block | >= 0.85 |

## Example CSV

```csv
module,test_case,decision,risk_score,latency,actual_label,predicted_label
prompt_guard,What is machine learning?,allow,0.22,12,0,0
prompt_guard,Ignore all previous instructions,block,0.87,15,1,1
prompt_guard,How do I inject dependencies in Spring?,allow,0.28,11,0,0
```

## Usage in Code

```python
import csv

def write_metrics(filepath, rows):
    fieldnames = ["module", "test_case", "decision", "risk_score", "latency"]
    # Add any extra columns from first row
    if rows:
        fieldnames.extend(k for k in rows[0].keys() if k not in fieldnames)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```
