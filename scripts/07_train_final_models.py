"""Train and save the final model of every project (random forest, the best model in the comparison).

Protocol (same as the validated experiments): train on the second-to-last release, keep the LAST release as the
'new release' to be scored in next step. The decision threshold comes from out-of-fold training predictions.
Writes models/<project>.joblib (one per project) and results/final_models.csv.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd
from src import config as C, data as D
from src.score import train_final, save_model

C.MODELS.mkdir(exist_ok=True); C.RESULTS.mkdir(exist_ok=True)
df = D.load_jira(); feats = D.feature_columns(df); rows = []
for i, (project, releases) in enumerate(C.RELEASE_ORDER.items(), 1):
    train_rel, new_rel = releases[-2], releases[-1]
    tr = df[(df.project == project) & (df.release == train_rel)]
    art = train_final(tr, feats, project=project, train_release=train_rel, scored_release=new_rel)
    path = C.MODELS / f"{project}.joblib"
    save_model(art, path)
    rows.append(dict(project=project, train_release=train_rel, new_release=new_rel, n_train=art["n_train"],
                     n_defective=art["n_pos"], threshold=round(art["threshold"], 2), model=art["model_name"]))
    print(f"[{i}/{len(C.RELEASE_ORDER)}] saved {path}", flush=True)
summary = pd.DataFrame(rows); summary.to_csv(C.RESULTS / "final_models.csv", index=False)
print(summary.to_string(index=False))
