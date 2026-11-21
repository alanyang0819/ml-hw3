from pathlib import Path
import gzip
import struct
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path(__file__).resolve().parents[2]
MNIST_DIR = ROOT / "data" / "mnist"
SPOOKY_DIR = ROOT / "data" / "spooky"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MAX_EPOCHS = 40

ARCHITECTURES = [
    (64,),
    (64, 64),
    (64, 64, 64),
    (128, 64),
    (128, 64, 32, 16),
    (64, 64, 64, 64),
]


def load_mnist_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: magic={magic}")
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(count, rows * cols)


def load_mnist_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: magic={magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def make_mlp(hidden_layers, max_epochs=MAX_EPOCHS):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=max_epochs,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=SEED,
    )


def make_logistic():
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=SEED,
    )


def fit_mlp_with_probe(dataset, X_train, y_train, X_val, y_val, layers):
    model = make_mlp(layers)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    probe_epochs = np.arange(1, len(model.loss_curve_) + 1)
    first_updates = []
    last_updates = []
    previous_weights = None

    probe_model = make_mlp(layers)
    probe_model.warm_start = True
    probe_model.max_iter = 1
    probe_model.early_stopping = False
    probe_model.tol = 0.0

    previous_weights = None
    for _ in probe_epochs:
        probe_model.fit(X_train, y_train)
        if previous_weights is not None:
            first_update = np.linalg.norm(
                probe_model.coefs_[0] - previous_weights[0]
            )
            last_update = np.linalg.norm(
                probe_model.coefs_[-1] - previous_weights[-1]
            )
            first_updates.append(float(first_update))
            last_updates.append(float(last_update))
        previous_weights = [
            weights.copy() for weights in probe_model.coefs_
        ]
        if len(first_updates) >= len(probe_epochs) - 1:
            break

    padded_first = [np.nan] + first_updates
    padded_last = [np.nan] + last_updates
    probe_df = pd.DataFrame({
        "epoch": probe_epochs,
        "train_loss": model.loss_curve_,
        "first_layer_weight_update_norm": padded_first[:len(probe_epochs)],
        "last_layer_weight_update_norm": padded_last[:len(probe_epochs)],
    })
    probe_df.to_csv(
        OUTPUT_DIR / f"{dataset.lower()}_{'_'.join(map(str, layers))}_gradient_proxy.csv",
        index=False,
    )

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)
    return model, {
        "dataset": dataset,
        "partition": "80/20 holdout",
        "model": "MLP",
        "hidden_layers": str(layers),
        "epochs": model.n_iter_,
        "accuracy": accuracy_score(y_val, predictions),
        "f1_macro": f1_score(y_val, predictions, average="macro"),
        "log_loss": log_loss(y_val, probabilities, labels=model.classes_),
        "training_time_seconds": elapsed,
        "seed": SEED,
    }


def fit_logistic(dataset, X_train, y_train, X_val, y_val, tfidf=False):
    model = make_logistic()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)
    return model, {
        "dataset": dataset,
        "partition": "80/20 holdout",
        "model": "LogisticRegression",
        "hidden_layers": "N/A",
        "epochs": "N/A",
        "accuracy": accuracy_score(y_val, predictions),
        "f1_macro": f1_score(y_val, predictions, average="macro"),
        "log_loss": log_loss(y_val, probabilities, labels=model.classes_),
        "training_time_seconds": elapsed,
        "seed": SEED,
    }


def run_cross_validation(dataset, X, y, hidden_layers=None, tfidf=False):
    if tfidf:
        estimator = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )),
            ("model", make_mlp(hidden_layers) if hidden_layers else make_logistic()),
        ])
    else:
        estimator = make_mlp(hidden_layers) if hidden_layers else make_logistic()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    start = time.time()
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1_macro", "neg_log_loss"],
        n_jobs=1,
        error_score="raise",
        return_train_score=True,
    )
    elapsed = time.time() - start
    result = {
        "dataset": dataset,
        "partition": f"{CV_FOLDS}-fold CV",
        "model": "MLP" if hidden_layers else "LogisticRegression",
        "hidden_layers": str(hidden_layers) if hidden_layers else "N/A",
        "accuracy": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "f1_macro": scores["test_f1_macro"].mean(),
        "f1_macro_std": scores["test_f1_macro"].std(),
        "log_loss": -scores["test_neg_log_loss"].mean(),
        "training_accuracy": scores["train_accuracy"].mean(),
        "training_time_seconds": elapsed,
        "seed": SEED,
    }
    print(dataset, "CV:", result)
    return result


def plot_loss_curves(dataset, probe_files):
    fig, ax = plt.subplots(figsize=(11, 7))
    for probe_file in probe_files:
        frame = pd.read_csv(probe_file)
        label = probe_file.stem.rsplit("_gradient_proxy", 1)[0].split("_", 1)[1]
        ax.plot(frame["epoch"], frame["train_loss"], marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"{dataset}: Training Loss Curves")
    ax.legend(title="Hidden Layers")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{dataset.lower()}_training_loss_curves.png")
    plt.close(fig)


def plot_gradient_proxy(dataset, probe_files):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for probe_file in probe_files:
        frame = pd.read_csv(probe_file)
        label = probe_file.stem.rsplit("_gradient_proxy", 1)[0].split("_", 1)[1]
        axes[0].plot(
            frame["epoch"],
            frame["first_layer_weight_update_norm"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            frame["epoch"],
            frame["last_layer_weight_update_norm"],
            marker="o",
            label=label,
        )
    axes[0].set_title("First hidden layer update norm")
    axes[1].set_title("Output layer update norm")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("L2 weight update")
        axis.grid(True)
    axes[1].legend(title="Hidden Layers", fontsize=8)
    fig.suptitle(
        f"{dataset}: gradient vanishing proxy "
        "(decreasing weight-update norms across epochs)"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{dataset.lower()}_gradient_proxy.png")
    plt.close(fig)


def run_mnist():
    train_images = MNIST_DIR / "train-images-idx3-ubyte.gz"
    train_labels = MNIST_DIR / "train-labels-idx1-ubyte.gz"
    test_images = MNIST_DIR / "t10k-images-idx3-ubyte.gz"
    test_labels = MNIST_DIR / "t10k-labels-idx1-ubyte.gz"

    X_full = load_mnist_images(train_images).astype(np.float32) / 255.0
    y_full = load_mnist_labels(train_labels).astype(str)
    X_test = load_mnist_images(test_images).astype(np.float32) / 255.0
    y_test = load_mnist_labels(test_labels).astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_full,
    )

    results = []
    models = {}
    probe_files = []
    for layers in ARCHITECTURES:
        model, result = fit_mlp_with_probe(
            "MNIST", X_train, y_train, X_val, y_val, layers
        )
        results.append(result)
        models[layers] = model
        probe_files.append(
            OUTPUT_DIR / f"mnist_{'_'.join(map(str, layers))}_gradient_proxy.csv"
        )

    plot_loss_curves("MNIST", probe_files)
    plot_gradient_proxy("MNIST", probe_files)

    _, logistic_result = fit_logistic(
        "MNIST", X_train, y_train, X_val, y_val
    )
    results.append(logistic_result)

    best_layers = max(
        models,
        key=lambda layers: accuracy_score(
            y_val, models[layers].predict(X_val)
        ),
    )
    print("MNIST selected architecture:", best_layers)

    results.append(run_cross_validation(
        "MNIST", X_full, y_full, hidden_layers=best_layers
    ))
    results.append(run_cross_validation(
        "MNIST", X_full, y_full, hidden_layers=None
    ))

    final_model = make_mlp(best_layers)
    final_start = time.time()
    final_model.fit(X_full, y_full)
    final_elapsed = time.time() - final_start
    predictions = final_model.predict(X_test)
    probabilities = final_model.predict_proba(X_test)
    test_result = {
        "dataset": "MNIST",
        "partition": "official test set",
        "model": "MLP",
        "hidden_layers": str(best_layers),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "log_loss": log_loss(y_test, probabilities, labels=final_model.classes_),
        "training_time_seconds": final_elapsed,
        "seed": SEED,
    }
    results.append(test_result)
    pd.DataFrame([test_result]).to_csv(
        OUTPUT_DIR / "mnist_official_test_results.csv", index=False
    )
    print("MNIST official test:", test_result)
    return results


def run_spooky():
    spooky_data = pd.read_csv(SPOOKY_DIR / "train.csv")
    X = spooky_data["text"]
    y = spooky_data["author"]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    results = []
    models = {}
    probe_files = []
    for layers in ARCHITECTURES:
        model, result = fit_mlp_with_probe(
            "Spooky", X_train_vec, y_train, X_val_vec, y_val, layers
        )
        results.append(result)
        models[layers] = model
        probe_files.append(
            OUTPUT_DIR / f"spooky_{'_'.join(map(str, layers))}_gradient_proxy.csv"
        )

    plot_loss_curves("Spooky", probe_files)
    plot_gradient_proxy("Spooky", probe_files)

    _, logistic_result = fit_logistic(
        "Spooky", X_train_vec, y_train, X_val_vec, y_val
    )
    results.append(logistic_result)

    best_layers = max(
        models,
        key=lambda layers: accuracy_score(
            y_val, models[layers].predict(X_val_vec)
        ),
    )
    print("Spooky selected architecture:", best_layers)

    results.append(run_cross_validation(
        "Spooky", X_train, y_train,
        hidden_layers=best_layers, tfidf=True,
    ))
    results.append(run_cross_validation(
        "Spooky", X_train, y_train,
        hidden_layers=None, tfidf=True,
    ))
    return results

def main():
    mnist_results = run_mnist()
    spooky_results = run_spooky()
    results = mnist_results + spooky_results
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "results.csv", index=False)
    print("All experiments completed. Results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
