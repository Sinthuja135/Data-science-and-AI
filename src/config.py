"""
Central configuration: paths, seeds, release order and feature groups.
Every other module in this package reads its settings from here
"""
import os
from pathlib import Path

# filesystem layout
ROOT = Path(__file__).resolve().parents[1]  # the project's top-level folder
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "raw"))
JIRA_DIR = DATA_DIR / "JIRA-defect-dataset"  # 32 release CSVs -- the primary dataset

# Where outputs get written. Created on demand by whichever script needs them
RESULTS = ROOT / "results"   # every CSV table used in the reports
FIGS = ROOT / "figures"      # every PNG figure used in the reports
MODELS = ROOT / "models"     # trained model artifacts (models/<project>.joblib), one per project

# reproducibility
SEED = 42   # fixed everywhere a random number is needed

# labels and feature groups
# The JIRA benchmark ships two different notions of "this file had a defect":
LABEL_REAL = "RealBug"   # a defect that actually affected THIS release (treated as ground truth)
LABEL_HEU = "HeuBug"     # a cheaper time-window heuristic guess at the same thing (noisier)

# Columns that describe the file/label bookkeeping rather than a predictive metric
NON_FEATURES = ["File", "HeuBug", "HeuBugCount", "RealBug", "RealBugCount"]

# Of the 65 JIRA metric columns, these two groups are about the file's *history*, not its current code
# (the remaining 54 are static code metrics like complexity, size, coupling, ...).
PROCESS = ["COMM", "ADEV", "DDEV", "Added_lines", "Del_lines"] # how much/who changed it recently
OWNERSHIP = ["OWN_LINE", "OWN_COMMIT", "MINOR_COMMIT", "MINOR_LINE", "MAJOR_COMMIT", "MAJOR_LINE"]

# Every release of every project, in chronological order. release_pairs() (in data.py) walks consecutive
# entries in each list to build "train on the past, test on the future" splits.
RELEASE_ORDER = {
    "activemq": ["5.0.0", "5.1.0", "5.2.0", "5.3.0", "5.8.0"],
    "camel": ["1.4.0", "2.9.0", "2.10.0", "2.11.0"],
    "derby": ["10.2.1.6", "10.3.1.4", "10.5.1.1"],
    "groovy": ["1_5_7", "1_6_BETA_1", "1_6_BETA_2"],
    "hbase": ["0.94.0", "0.95.0", "0.95.2"],
    "hive": ["0.9.0", "0.10.0", "0.12.0"],
    "jruby": ["1.1", "1.4.0", "1.5.0", "1.7.0.preview1"],
    "lucene": ["2.3.0", "2.9.0", "3.0.0", "3.1"],
    "wicket": ["1.3.0-incubating-beta-1", "1.3.0-beta2", "1.5.3"],
}

# thresholds
# a train/test release pair is skipped unless BOTH sides have at least this many
# precision/recall numbers are too noisy to trust
MIN_POS = 5          
# when two metrics' Spearman correlation exceeds this, one is dropped
CORR_THRESHOLD = 0.9 
