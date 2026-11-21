# Machine Learning Homework 3

Run both MNIST and Spooky Author Identification experiments:

```bash
uv run ml
```

The program writes reproducible results to `results/`:

- `results.csv`: holdout, cross-validation, and MNIST official-test metrics
- `report.md`: mini-report template populated with the current experiment tables
- `mnist_training_loss_curves.png`, `spooky_training_loss_curves.png`
- `mnist_gradient_proxy.png`, `spooky_gradient_proxy.png`
- per-architecture gradient-proxy CSV files

The gradient plots use first/last-layer weight-update norms as an evidence proxy,
because scikit-learn's `MLPClassifier` does not expose per-epoch gradients.