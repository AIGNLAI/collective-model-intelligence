# Reproduction guide

[Project home](../README.md) · [Experimental results](experimental-results.md)

## Reproduce report Section 6.3

The recorded baseline environment is **Python 3.12.2 on Linux with CUDA 12.1**.
All dependency declarations live in `pyproject.toml`: `.[data,orbit]` supports the general
quick start, while `.[report]` installs the fixed report versions in a separate environment.

```sh
python3.12 -m venv .venv-report
source .venv-report/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[report]'
python -m examples.reproduce_report --mode replay --output runs/report-replay
python -m examples.reproduce_report --mode fit --device cuda:0 --output runs/report-refit
```

Replay downloads SHA-256-checked predictions from the pinned dataset and recomputes
**65 runs across 13 methods**, including Mio. It requires no GPU or model-service credentials.
Once the dataset and predictions are cached, `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`
allows replay without a network connection.
Fit retrains the **12 public baselines**; Mio remains replay-only. Use `--device cpu` for CPU
fitting. Hardware and library differences can affect fitted neural outputs; refitting is
not promised to be bitwise identical. Commands refuse to overwrite an existing output directory.

All methods use the released embeddings: BGE-M3 for Text/Chinese and 1,024-dimensional
CLIP text/image fusion for Multimodal. No encoder weights or media downloads are needed.
Data and frozen predictions are hosted in the [pinned dataset release](https://huggingface.co/datasets/SinapisAI/RobustRouteBench/tree/e05b897bf3b92bc9913409959a150f75cecbb427).
Mio's internal trainer, the Model Eval implementation, and production Mos services are not released.
The separate [Model Eval experimental settings](model-eval.md) document the characterization studies.

The replay command compares the complete aggregate against the release's
[reference results](https://huggingface.co/datasets/SinapisAI/RobustRouteBench/blob/e05b897bf3b92bc9913409959a150f75cecbb427/report/aggregate-results.json)
and stops on a mismatch. It writes per-run evaluations, an aggregate, and the data revision
to the chosen output directory under `runs/`.

## Split discipline

Data revision: `e05b897bf3b92bc9913409959a150f75cecbb427`, release `v1.1.2-20261015`.
Each 500-row validation split (400 standard + 100 robust) remains in the published test split.
After tuning, final scoring must exclude **every validation source group and its linked variants**.
The reproduction commands apply this exclusion; the basic loader returns splits unchanged.

Source grouping uses `base_query_id` when present, otherwise `query_id`, for Text/Chinese,
and `source_sample_id` for Multimodal. Predictions are aligned by query ID and candidate-model
ID. Test labels, realized costs, and oracle choices stay outside fitting and inference.

| Scenario | Published test | Validation | Excluded with linked variants | Final standard | Final robust |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 7,188 | 500 | 606 | 3,688 | 2,894 |
| Multimodal | 7,000 | 500 | 500 | 4,600 | 1,900 |
| Chinese | 6,500 | 500 | 600 | 3,900 | 2,000 |

The final population is **18,982 requests**. Multimodal standard and robust populations are
independently sampled; their difference is not a paired perturbation effect. Available metadata
does not establish document- or repository-level disjointness. The corrected release restores
only provenance-confirmed Text score regressions; failed requests remain zero, and other
legacy outcomes were not independently regraded.

## Metric compatibility

The quick start and report reproduction use `ReportBudgetCurveEvaluator`:

- Select by predicted utility under predicted-cost feasibility on a common 100-point nominal-budget grid.
- Compute nAUC by integrating cumulative-maximum quality over **nominal budget**, then divide by the budget interval.
- Never insert fixed-model reference points. `Cost@τ` is the minimum realized cost among router points reaching the target; unreachable targets are N/R.
- For the three Multimodal fits, average aligned curves **pointwise before** computing metrics. Standard errors enumerate all 27 ordered run-bootstrap resamples.
- Macro nAUC equally averages the six standard/robust split values.

The shared `BudgetCurveEvaluator` remains an internal base with legacy realized-cost Pareto
AUC; use the report evaluator for comparisons. Its predicted-cheapest fallback can exceed a
nominal budget. Stored latency is a historical outcome, not a live endpoint guarantee.

## Cost and score units

| Artifact | Convention |
| --- | --- |
| Public RobustRouteBench arrays | CNY; historical USD proxies converted at **1 USD = 7.2 CNY** |
| Report §6.3 tables and frozen results | Text/Multimodal use USD proxy units; Chinese uses CNY |
| Mio public-task tracker (§6.1) | Total CNY divided by method-specific evaluation count; source conversion **1 USD = 7 CNY** |

Mio public-task means equally weight nine normalized scores and per-request costs;
MT-Bench scores are scaled from 0–10 to 0–100. The report's method-specific request
counts apply before averaging. The two exchange-rate conventions belong to separate studies.
