# Stage 2 retraining

`train_stage2.py` is the isolated protocol runner for the ten preselected
configurations. It writes only below this directory and refuses to overwrite a
non-empty seed directory.

Run the required preflight check with:

```powershell
python training_experiments/train_stage2.py --sanity
```

The complete sequential protocol is started only with:

```powershell
python training_experiments/train_stage2.py --run-all
```

No checkpoint from `outputs/final_model` is reused or modified.
