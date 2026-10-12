<h1 align="center">P3IBench</h1>


<p align="center">
  <strong>A Benchmark for Proactive Video Understanding</strong>
</p>

<p align="center">
  <a href="?">
    <img src="?" align="center" alt="arXiv"> <strong>Paper</strong>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="?">
    <img src="?" width="20" align="center" alt="Homepage"> <strong>Homepage</strong>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="?">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="24" align="center" alt="Hugging Face"> <strong>Models &amp; Data</strong>
  </a>
</p>

## News

- The inference and evaluation scripts are available. Paper, project page, and dataset download links will be added upon release.

## Quick Start

### Installation

Run the following commands from the repository root (the directory containing `main.py`). Python 3.12, a GPU suitable for vLLM, and model weights are required. Install dependencies in a dedicated Conda environment:

```bash
conda create -n P3IBench python=3.12 -y
conda activate P3IBench
python -m pip install -r requirements.txt
mkdir -p train data output
```

### Download the Benchmark

**Dataset URL: coming soon.** Once the dataset is published, replace `<DATA_ARCHIVE_URL>` with the official download URL and extract it into `data/`:

```bash
curl -L "<DATA_ARCHIVE_URL>" -o data/benchmark.tar.gz
tar -xzf data/benchmark.tar.gz -C data
```

The archive should place dataset directories directly under `data/` (move them up one level if the archive contains an extra enclosing directory):

```text
data/
├── Prediction_Accident/
│   ├── Prediction_Accident.json
│   └── videos/
├── Perception_Object/                 # Perception_Object.json and videos/
├── Prediction_Crime/                  # Prediction_Crime.json and videos/
├── Perception_Event_atomic/           # Perception_Event_atomic.json and videos/
├── Perception_Event_compositional/    # Perception_Event_compositional.json and videos/
├── Perception_Event_eventtype/        # Perception_Event_eventtype.json and videos/
└── Intervention/                      # Intervention.json and video/
```

Each annotation JSON must contain a nonempty top-level `videos` array. Entries require `question_id`, `video`, and `duration`; evaluation also uses the reference `response` fields. Video paths in annotations are relative to their dataset's `videos/` or `video/` directory. The diagram describes the expected layout, **not** a replacement for the official annotations. For data stored elsewhere, pass `--data-root /path/to/data` to `main.py`.

Dataset directories and primary annotation filenames use the names above. Original `question_id` prefixes are preserved for task routing and prediction matching; do not rename sample IDs or video files.

### Start a vLLM Backend

In the first terminal, start an OpenAI-compatible vLLM service. Replace the model ID or path with the model you want to evaluate:

```bash
conda activate P3IBench
vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --served-model-name Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 --port 8010
```

### Run Inference and Standard Evaluation

In a second terminal, from the repository root:

```bash
conda activate P3IBench
python main.py --port 8010

# Or run only selected datasets:
python main.py --port 8010 --datasets Prediction_Accident Perception_Event_atomic
```

By default, `main.py` runs Prediction_Accident, Perception_Object, Prediction_Crime, Perception_Event_atomic, Perception_Event_compositional, Perception_Event_eventtype, and Intervention in sequence. It discovers the served model via `/v1/models`; alternatively, provide `--model MODEL_NAME` or `--api-base http://HOST:PORT/v1` for a remote backend. Other options include `--fps`, `--parallelism`, `--max-frames`, and `--request-retries` (see `python main.py --help`). Each dataset receives default, first-only, bidirectional default, and bidirectional first-only scores. If a dataset fails, the script records the error, continues with the remaining datasets, and exits with a nonzero status at the end.

## Semantic Evaluation

Semantic scoring is **not** run by `main.py`. Start a separate OpenAI-compatible text-judge backend.

```bash
vllm serve <JUDGE_MODEL_PATH_OR_ID> \
  --served-model-name Qwen3-32B-FP8 \
  --host 127.0.0.1 --port 8011
```

After inference, use its run ID and a dataset output directory. For example, score Intervention without overwriting the standard results:

```bash
RUN_ID=<YYYYMMDD-HHMMSS>
INPUT_DIR="output/$RUN_ID/Intervention"
python eval/unified.py --input-dir "$INPUT_DIR" \
  --mode default --base-score semantic \
  --semantic-model Qwen3-32B-FP8 \
  --semantic-api-base http://127.0.0.1:8011/v1 \
  --details "$INPUT_DIR/score/semantic_default.json" \
  --summary "$INPUT_DIR/score/semantic_default_summary.json"
```

## Output Structure

Each invocation creates `output/<YYYYMMDD-HHMMSS>/` (UTC+8 timestamp). Set `--output-root` to change the output location; dataset directories are placed directly inside that run directory.

```text
output/<RUN_ID>/
├── Docs/
│   ├── aggregate_summary.json       # Status and scores for selected datasets
│   ├── README.md                    # Per-run summary table
│   └── *.log                        # Inference and evaluation logs
├── Prediction_Accident/              # One directory per selected dataset
│   ├── pred.jsonl                    # Predictions
│   ├── inputs/annotation.json        # Annotation copy used in this run
│   └── score/                        # Detailed scores and *_summary.json files
└── <DATASET>/                        # Other selected datasets follow the same layout
```

Consult the dataset directories inside `output/<RUN_ID>/` for actual results; `Docs/` is the report generated **inside that run**, not a separate archive of historical results.

## Repository Layout

```text
main.py             # Inference, standard evaluation, and aggregation
infer/              # Streaming inference and supporting modules
eval/               # Standard, compositional, and semantic evaluators
train/              # Training code
data/               # Downloaded dataset
output/             # Generated predictions and scores
requirements.txt
```

## Citation

The paper and BibTeX entry will be provided when the arXiv link becomes available.
