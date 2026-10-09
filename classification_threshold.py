from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50
DEFAULT_COST_FP = 1.0
DEFAULT_COST_FN = 10.0
TARGET = "target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подбор порога бинарной классификации по цене ошибок."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("data.csv"),
        help="Путь к CSV с целевой колонкой target.",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=DEFAULT_COST_FP,
        help="Цена ложноположительной ошибки FP.",
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=DEFAULT_COST_FN,
        help="Цена ложноотрицательной ошибки FN.",
    )
    return parser.parse_args()


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def error_cost(
    y_true: pd.Series,
    y_probability: np.ndarray,
    threshold: float,
    cost_fp: float,
    cost_fn: float,
) -> tuple[float, int, int, int, int]:
    y_pred = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = fp * cost_fp + fn * cost_fn
    return float(total_cost), int(tn), int(fp), int(fn), int(tp)


def choose_threshold(
    y_true: pd.Series,
    y_probability: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []

    # Порог выбирается только на validation, а не на test.
    for threshold in np.arange(0.01, 1.00, 0.01):
        threshold = round(float(threshold), 2)
        total_cost, tn, fp, fn, tp = error_cost(
            y_true, y_probability, threshold, cost_fp, cost_fn
        )
        rows.append(
            {
                "threshold": threshold,
                "cost": total_cost,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    search = pd.DataFrame(rows)
    best_row = search.sort_values(
        by=["cost", "fn", "fp", "threshold"],
        ascending=[True, True, True, True],
    ).iloc[0]

    return float(best_row["threshold"]), search


def classification_metrics(
    y_true: pd.Series,
    y_probability: np.ndarray,
    threshold: float,
    cost_fp: float,
    cost_fn: float,
) -> dict[str, float | int]:
    y_pred = (y_probability >= threshold).astype(int)
    total_cost, tn, fp, fn, tp = error_cost(
        y_true, y_probability, threshold, cost_fp, cost_fn
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "business_cost": total_cost,
    }


def save_curves(
    y_true: pd.Series,
    y_probability: np.ndarray,
    output_dir: Path,
) -> tuple[float, float, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_probability)
    roc_auc = roc_auc_score(y_true, y_probability)

    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    pr_auc = auc(recall, precision)
    average_precision = average_precision_score(y_true, y_probability)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="Случайный классификатор")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC-кривая")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    positive_share = float(np.mean(y_true))
    plt.figure(figsize=(7, 5))
    plt.plot(
        recall,
        precision,
        label=f"PR AUC = {pr_auc:.3f}; AP = {average_precision:.3f}",
    )
    plt.axhline(
        positive_share,
        linestyle="--",
        label=f"Базовый precision = {positive_share:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall кривая")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close()

    return float(roc_auc), float(pr_auc), float(average_precision)


def main() -> None:
    args = parse_args()

    if args.cost_fp < 0 or args.cost_fn < 0:
        raise ValueError("Цена ошибки не может быть отрицательной.")

    df = pd.read_csv(args.data)
    if TARGET not in df.columns:
        raise ValueError(f"В данных должна быть целевая колонка '{TARGET}'.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # 60% train, 20% validation, 20% test.
    # Validation нужен для выбора порога, test остается честным финальным контролем.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    model = build_model()
    model.fit(X_train, y_train)

    # Используем вероятности положительного класса, как требует задание.
    val_probability = model.predict_proba(X_val)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]

    selected_threshold, threshold_search = choose_threshold(
        y_val,
        val_probability,
        args.cost_fp,
        args.cost_fn,
    )

    default_metrics = classification_metrics(
        y_test,
        test_probability,
        DEFAULT_THRESHOLD,
        args.cost_fp,
        args.cost_fn,
    )
    selected_metrics = classification_metrics(
        y_test,
        test_probability,
        selected_threshold,
        args.cost_fp,
        args.cost_fn,
    )

    roc_auc, pr_auc, average_precision = save_curves(
        y_test,
        test_probability,
        Path("plots"),
    )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(
        [
            {"variant": "default_0.5", **default_metrics},
            {"variant": "selected", **selected_metrics},
        ]
    )
    comparison.to_csv(results_dir / "metrics.csv", index=False)
    threshold_search.to_csv(results_dir / "threshold_search.csv", index=False)

    print(f"Размеры: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")
    print(
        f"Цена ошибок: FP={args.cost_fp:g}, FN={args.cost_fn:g} "
        f"(FN в {args.cost_fn / args.cost_fp:.1f} раз дороже FP)"
        if args.cost_fp > 0
        else f"Цена ошибок: FP={args.cost_fp:g}, FN={args.cost_fn:g}"
    )
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Подобранный на validation порог: {selected_threshold:.2f}")
    print()
    print("Сравнение на test:")
    print(
        comparison[
            [
                "variant",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "fp",
                "fn",
                "business_cost",
            ]
        ].to_string(index=False)
    )

    default_cost = float(default_metrics["business_cost"])
    selected_cost = float(selected_metrics["business_cost"])
    print()
    if selected_cost < default_cost:
        print(
            f"Вывод: порог {selected_threshold:.2f} уменьшил цену ошибок "
            f"с {default_cost:.0f} до {selected_cost:.0f}."
        )
    elif selected_cost == default_cost:
        print(
            f"Вывод: порог {selected_threshold:.2f} дал ту же цену ошибок "
            f"({selected_cost:.0f}), но изменил баланс precision/recall."
        )
    else:
        print(
            f"Вывод: на test цена ошибок выросла с {default_cost:.0f} "
            f"до {selected_cost:.0f}; порог всё равно выбирался только на validation, "
            "чтобы не подгоняться под test."
        )


if __name__ == "__main__":
    main()
