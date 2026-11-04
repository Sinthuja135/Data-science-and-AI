# Data-science-and-AI
Explainable Cross Release Software Defect Prediction on the JIRA Defect Dataset

## Quick start

https://zenodo.org/api/records/6342328 
Large Defect Prediction Benchmark v1.0 (DOI 10.5281/zenodo.6342328)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# put the benchmark data under data/raw

# run below commands
python scripts/01_eda.py 
python scripts/02_run_models.py 
python scripts/02b_model_improvements.py
python scripts/03_summarise.py
```