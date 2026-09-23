import json
import pathlib

root = pathlib.Path("ssl_pretraining_experiment/supervised")
for s in range(5):
    p = root / f"seed{s}"
    log = json.loads((p / "training_log.json").read_text())
    best = max(log["val_auc"])
    best_epoch = log["val_auc"].index(best) + 1
    files = [f.name for f in p.iterdir()]
    print(f"seed{s}: epochs={len(log['val_auc'])}, best_val_auc={best:.4f}, best_epoch={best_epoch}, files={files}")
