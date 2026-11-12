# EnGrade-CP

Leakage-audited ordinal grading of enhancer activity (non-enhancer < weak < strong) with a frozen genomic-language-model view, gated multi-view fusion and tier-conditional conformal prediction sets. This is the code that produced every number, table and figure in the accompanying manuscript.

## Run on Google Colab (free tier)
1. Upload `EnGrade_CP_Colab.ipynb` to Colab (File > Upload notebook).
2. Runtime > Change runtime type > T4 GPU (a CPU-only runtime also works, just more slowly).
3. Runtime > Run all.

Everything is downloaded automatically:
- Benchmark: iEnhancer-2L training set and iEnhancer-EL independent set, as released with Enhancer-FRL
  (`training_testing_datasets.rar` from github.com/wangchao-malab/Enhancer-FRL).
- External set: Genomic Benchmarks `human_enhancers_cohn` (Hugging Face Hub, `katarinagresova/Genomic_Benchmarks_human_enhancers_cohn`).
- Frozen genomic language model: DNABERT-6 (`zhihan1996/DNA_bert_6`).

Outputs go to `engrade_outputs/` and are zipped to `engrade_outputs.zip`.

## Local run
```
pip install -r requirements.txt
python engrade_colab.py
```

## Configuration
Edit the `CFG` dictionary in the configuration cell:
- `quick=True` runs a 2-fold, 1-seed smoke test (about 20 minutes on CPU).
- `ext_n` sets how many external sequences are scored (6948 = the whole external test split).
- `use_glm=False` drops the DNABERT view. The notebook also turns it off on its own if the model cannot be downloaded.

## Output files
| File | Content |
|---|---|
| T1_leakage_audit.csv | duplicates, 8-mer containment, abutting tiles, loci, label purity |
| T2_cv_random_vs_blocked.csv | all metrics, random and chromosome-blocked protocols, bootstrap CIs |
| T3_random_minus_blocked.csv | protocol differences |
| T4_independent_legacy_vs_clean.csv | legacy (duplicated) vs clean independent evaluation |
| T5_ablation_blocked.csv | component ablation (seed 0, mean and SD over folds) |
| T6_paired_bootstrap.csv | EnGrade-CP minus each baseline, paired bootstrap |
| T7_conformal_alpha0.1.csv | marginal vs tier-conditional sets, all models |
| T8_engrade_coverage_vs_alpha.csv | coverage and set size vs alpha |
| T9_selective_prediction.csv | error on accepted singleton sets |
| T10_external_cohn.csv | zero-shot external AUC, MCC, coverage |
| T11_attributed_6mers_strong.csv | gradient x input attribution, enriched 6-mers |
| T12_gate_weights_by_tier.csv | mean gate weight per view and tier |
| T13_engrade_set_composition.csv | prediction-set composition by true tier |
| T14_robustness_substitution.csv | metrics under point substitutions |
| T15_efficiency.csv | parameters, training time, scoring and embedding latency |
| T16_per_fold.csv | per held-out chromosome group results |
| T17_confusion_engrade.csv | three-tier confusion matrix |
| Fig1–Fig10 *.png | black-and-white manuscript figures (600 dpi) |
| predictions.pkl | all out-of-fold, independent, external and calibration predictions |

## Label convention for the independent set
The independent FASTA carries no labels. Following the iEnhancer-EL convention (also used in Enhancer-FRL), sequences 1–100 are strong enhancers, 101–200 are weak enhancers and 201–400 are non-enhancers. The notebook checks that the first 200 sequences match `strong_weak_testing.fasta`.

## Reference run
Executed end to end as a notebook on a 2-core CPU (Python 3.12, PyTorch 2.15, transformers 5.22, scikit-learn 1.9). A T4 GPU runtime is faster because the DNABERT embedding and network training run on the GPU.
