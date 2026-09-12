"""Pipeline complet, corrige : split par client, sans la feature de montant
biaisee, avec comparaison baseline/accuracy/recall/precision/F1, et une
fonction d'inference reutilisable."""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

from radar_risque_impaye import generate_invoices, add_recent_late_rate, add_sector_gap
from radar_risque_impaye.features import build_feature_matrix

FEATURE_COLS = ["taux_retard_recent", "ecart_taux_secteur", "secteur"]

factures = generate_invoices(n_clients=60, seed=7)
factures = add_sector_gap(add_recent_late_rate(factures, window=5))
print("Nombre de factures apres feature engineering:", len(factures))

X, y, groups, cols = build_feature_matrix(factures, feature_cols=FEATURE_COLS)

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
print(
    "clients en train:",
    len(set(groups[train_idx])),
    "- clients en test:",
    len(set(groups[test_idx])),
)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)
baseline_pred = np.zeros_like(y_test)

print("\n=== Accuracy (metrique trompeuse ici) ===")
print("modele:", round(accuracy_score(y_test, predictions), 4))
print("baseline:", round(accuracy_score(y_test, baseline_pred), 4))

clf_balanced = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
clf_balanced.fit(X_train_scaled, y_train)
predictions_balanced = clf_balanced.predict(X_test_scaled)

print(
    "\n=== Metriques adaptees a un probleme desequilibre (class_weight='balanced') ==="
)
print("recall modele:", round(recall_score(y_test, predictions_balanced), 4))
print("recall baseline:", round(recall_score(y_test, baseline_pred), 4))
print("precision modele:", round(precision_score(y_test, predictions_balanced), 4))
print("f1 modele:", round(f1_score(y_test, predictions_balanced), 4))
print("f1 baseline:", round(f1_score(y_test, baseline_pred), 4))
print("matrice de confusion:\n", confusion_matrix(y_test, predictions_balanced))


def predict_facture_risque(taux_retard_recent, ecart_taux_secteur, secteur):
    secteur_cols = [c for c in cols if c.startswith("secteur_")]
    row = {c: 0 for c in secteur_cols}
    row[f"secteur_{secteur}"] = 1
    row["taux_retard_recent"] = taux_retard_recent
    row["ecart_taux_secteur"] = ecart_taux_secteur
    X_row = np.array([[row[c] for c in cols]])
    X_row_scaled = scaler.transform(X_row)
    return clf_balanced.predict_proba(X_row_scaled)[0, 1]


print("\n=== Inference sur deux cas types ===")
print("client BTP a risque:", round(predict_facture_risque(0.6, 0.3, "BTP"), 3))
print(
    "client Services fiable:", round(predict_facture_risque(0.0, -0.1, "Services"), 3)
)
