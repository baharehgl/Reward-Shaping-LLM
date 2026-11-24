# Stanford COVID-19 Wearables Phase 2: resting heart rate run

Runs the pipeline from arXiv 2601.02511 (`myasp-newdataset.py`) on the Stanford Phase 2 data.
Train only on the people in `train_ids.txt`, report only on the people in `test_ids.txt`
(anyone in neither list is skipped).

## Files
| File | What it does |
|---|---|
| `convert_stanford_p2.py` | Raw Phase 2 files -> hourly resting heart rate CSVs (`train/`, `test/`, `normal/`, `manifest.csv`) |
| `myasp-newdataset.py` | The model. Trains on `train/`, VAE on `normal/`, then predicts every piece in `test/` -> `results/predictions.csv` |
| `stanford_utils.py` | Writes the predictions (`predict_split`) |
| `evaluate_results.py` | `results/results_per_person.csv`, `summary.csv`, three PNG figures (300 dpi) |
| `run_stanford.sh` | Runs all three steps |

## How to run
```
export OPENAI_API_KEY=...        # required by llm_shaping.py even for llama-3 / phi-2
export LLM_CHOICE=llama-3        # or gpt-3.5-turbo, phi-2
./run_stanford.sh /path/to/COVID-19-Phase2-Wearables /path/to/stanford_p2_labels.csv
```
Needs the same environment as `myasp-A1.py` (TensorFlow, torch, transformers, openai, scikit-learn, scipy)
plus pandas and matplotlib. Optional: `ALERT_MODE=hour`, `NUM_EPISODES=3` (training episodes).
The converter and `evaluate_results.py` need only pandas, numpy and matplotlib.

## What the data is
- **value** = resting heart rate: a minute is resting if there were no steps in the previous 12 minutes
  (steps file); each hour = mean heart rate over its resting minutes; hours with fewer than 10 resting
  minutes are left out.
- **datetime** column keeps real time. It is the last column because `env.py` reads columns by position
  (timestamp, value, is_anomaly). A gap of more than 6 hours between valid hours starts a new piece
  (`<id>_p1.csv`, `<id>_p2.csv`, ...). Pieces shorter than 30 hours are dropped.
- **is_anomaly** = 1 from day 0 - 7 days to day 0 + 13 days (21 days). Day 0 = symptom onset, or the test
  date if onset is empty. People not in `stanford_p2_labels.csv` are treated as healthy (all 0).

## Choices to know about
- "Fewer than 10 resting minutes": resting minutes are counted by the steps rule even if the watch logged no
  heart rate in them (`--count-mode hr` is the stricter version).
- The first 25 hours of every piece are not scored (the model needs a 25-hour window).
- Training changes vs `myasp-A1.py`: all train people are used for training (`validation_separate_ratio = 1.0`),
  the VAE uses a random sample of 200,000 normal windows (`MAX_VAE_WINDOWS`), episodes can be set with `NUM_EPISODES`.
- Alert (default): one alert per run of consecutive flagged rows in a piece, placed at its first hour.
  `--alert-mode hour` counts every flagged hour. An alert is inside the illness window if it starts inside it.
- Point-adjust: if any hour of a true illness segment is flagged, the whole segment counts as detected.
- Stanford paper reference (symptomatic): 58 of 66 detected, 53 of them at or before onset. Our test set has
  fewer symptomatic people, so the denominators differ.
