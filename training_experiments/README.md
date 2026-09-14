# Stage 2 retraining

`train_stage2.py` is the isolated protocol runner for the ten preselected
configurations. It writes only below this directory and refuses to overwrite a
non-empty seed directory.

Run the required preflight check with:

```powershell
python training_experiments/train_stage2.py --sanity
```

Train one configuration without evaluation (recommended when GPU quota is
tight):

```powershell
python training_experiments/train_stage2.py --config A1S4V2
```

In a fresh GPU session, evaluate it separately:

```powershell
python training_experiments/evaluate_stage2.py --config A1S4V2
```

To distribute evaluation over three sessions, run `--dataset Internal`, then
`--dataset Diveshzz`, then `--dataset ThyroidPretrain`. The evaluator combines
the saved independent results only after all three datasets are complete.

No checkpoint from `outputs/final_model` is reused or modified.
