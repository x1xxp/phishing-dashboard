# PhishGuard — Phishing URL Risk Assessment Dashboard

An interactive Streamlit dashboard for **rule-based, explainable phishing URL risk
assessment**, designed and developed by **Abdalla**.

## Live demo

[Launch PhishGuard](https://phishguard-url-risk.streamlit.app/)

The application extracts structural features from a URL string (IP address used as a
hostname, Punycode, `@` symbol, URL-shortening services, URL length, subdomain depth,
suspicious keywords, misleading patterns, absence of HTTPS, and digit/hyphen ratio),
combines them with **frozen calibrated weights**, and maps the resulting weighted score
to a low / medium / high risk category using **frozen calibrated thresholds**. Every
triggered rule and its individual contribution are displayed, so the outcome is
transparent rather than a black-box prediction.

The system analyses the URL **as text only** — it never visits, requests, or renders a
submitted website.

## Features

- **URL Scanner** — score a single URL and see every triggered rule with its contribution.
- **Batch Scanner** — upload a CSV with a `url` column (and an optional `label` column) to
  score up to 10,000 URLs and download the results.
- **Model Insights** — the frozen calibrated test-set metrics and confusion matrix from
  `results.json`.
- **Methodology** — the scoring pipeline and the calibrated feature/weight reference table.
- **About** — project and submission details.

## Requirements

- Python 3.10 or newer (3.12 recommended)
- The packages pinned in `requirements.txt`

## Setup

### 1. Create a virtual environment

**Windows (PowerShell)**

```powershell
cd "C:\Users\x1xxp\Desktop\GRADTU CLOUDE\CODE\phishing-dashboard"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```bat
cd "C:\Users\x1xxp\Desktop\GRADTU CLOUDE\CODE\phishing-dashboard"
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
cd phishing-dashboard
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run dashboard.py
```

Streamlit prints a local URL (by default `http://localhost:8501`) — open it in a browser.
To use a different port:

```bash
streamlit run dashboard.py --server.port 8502
```

Press `Ctrl+C` in the terminal to stop the server.

> If the `streamlit` command is not found, use `python -m streamlit run dashboard.py`.

## Project structure

```
phishing-dashboard/
├── dashboard.py          # Streamlit application: UI, layout, charts, batch analysis
├── features.py           # Feature extraction and scoring logic (FROZEN — do not modify)
├── results.json          # Calibrated weights, thresholds, test metrics (FROZEN — do not modify)
├── requirements.txt      # Pinned runtime dependencies
├── README.md             # This file
├── .gitignore            # Excludes caches, virtual environments, logs, local settings
├── .streamlit/
│   └── config.toml       # Dark theme assumed by the dashboard styling
└── data/
    └── phiUSIIL.zip      # PhiUSIIL Phishing URL Dataset, retained for provenance
```

### Notes on the individual files

- **`dashboard.py`** resolves every project file from
  `BASE_DIR = Path(__file__).resolve().parent`, so the app runs correctly regardless of the
  terminal's current working directory. It imports the scoring functions from `features.py`
  and loads the calibration payload from `results.json`.
- **`features.py`** contains the frozen rule definitions, raw sub-score extraction, the
  weighted-score function, and the risk-category threshold logic.
- **`results.json`** contains the calibrated weights, the calibrated thresholds
  (`calibrated_low_high`, `calibrated_med_high`) and the reported test-set metrics. The
  dashboard reads these values; it never recomputes or overwrites them.
- **`data/phiUSIIL.zip`** is the source dataset archive kept for provenance and
  reproducibility. It is **not read at runtime** and must remain compressed and unmodified.

## Scoring model (frozen)

The score is a weighted sum of the raw sub-scores. Banded features (`url_length`,
`subdomain`, `keyword`, `special_pattern`) contribute `weight × (sub_score / 2)`; binary
features contribute `weight × sub_score`. The risk category is then:

| Category | Condition |
| --- | --- |
| Low | `score ≤ calibrated_low_high` |
| Medium | `calibrated_low_high < score ≤ calibrated_med_high` |
| High | `score > calibrated_med_high` |

The weights, thresholds, and reported metrics are fixed by the project's calibration and
must not be edited.

## Scope and limitations

PhishGuard evaluates the URL string only. It does not fetch webpages, inspect page
content, replace browser or antivirus protection, or query threat-intelligence services.
Its output is decision-support information, not proof that a site is malicious or safe.
