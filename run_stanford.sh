#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PHASE2_DIR="${1:?give the path to the COVID-19-Phase2-Wearables folder}"
LABELS="${2:?give the path to stanford_p2_labels.csv}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY first}"
export LLM_CHOICE="${LLM_CHOICE:-llama-3}"
ALERT_MODE="${ALERT_MODE:-run}"

echo "1/3 converting the data (train_ids.txt / test_ids.txt split; safe to re-run, it resumes)"
python3 convert_stanford_p2.py --root "$PHASE2_DIR" --labels "$LABELS" \
        --train train_ids.txt --test test_ids.txt --out datasets/STANFORD_P2_RHR

echo "2/3 training on train/ and predicting test/ with LLM_CHOICE=$LLM_CHOICE"
python3 myasp-newdataset.py

echo "3/3 results"
python3 evaluate_results.py --pred results/predictions.csv \
        --manifest datasets/STANFORD_P2_RHR/manifest.csv --out results --alert-mode "$ALERT_MODE"
