"""
evaluate.py
-----------
Evaluates a trained model: accuracy, precision, recall, F1, and a
confusion matrix plot saved to figures/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(artifacts: dict, save_plot: bool = True) -> dict:
    model = artifacts["model"]
    X_test_scaled = artifacts["X_test_scaled"]
    y_test = artifacts["y_test"]
    encoder = artifacts["encoder"]

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_, output_dict=True
    )
    report_text = classification_report(y_test, y_pred, target_names=encoder.classes_)

    print(f"Accuracy: {acc:.3f}\n")
    print(report_text)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Regime Classification - Confusion Matrix")
    plt.tight_layout()

    if save_plot:
        out_path = FIGURES_DIR / "confusion_matrix.png"
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved confusion matrix -> {out_path}")

    plt.close(fig)

    return {"accuracy": acc, "report": report, "predictions": y_pred}


if __name__ == "__main__":
    import pandas as pd
    from train_model import train

    features = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv",
        index_col=0, parse_dates=True,
    )
    artifacts = train(features)
    evaluate(artifacts)