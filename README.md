NAGPRA Repatriation Analysis

Analyzes Native American remains repatriation data under NAGPRA.

Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib statsmodels
```

Run

```bash
python3 build_nagpra_data.py  # Parse raw data → nagpra_data_full.csv
python3 script.py             # Analyze & visualize
```
