"""Small reproducible portfolio project using synthetic, non-sensitive data."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

def train(seed=42):
    x, y = make_classification(n_samples=1200, n_features=8, n_informative=5, weights=[0.85, 0.15], random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=seed)
    model = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=seed)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    return model, x_test, y_test, probabilities

def run():
    model, x, y, pred = train()
    print(f"ROC AUC: {roc_auc_score(y, pred):.3f}")

if __name__ == "__main__":
    run()
