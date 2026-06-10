This directory stores training proof artifacts generated automatically.

After running `python train.py`, the LSTM training loop saves:

  epoch_plot.png   — dual-axis plot of train loss & val ROC-AUC per epoch
                     (with best-epoch and early-stop markers)

These files are NOT committed to git (see .gitignore) so that large binary
artifacts do not bloat the repository.  Run the training pipeline locally
to regenerate them.
