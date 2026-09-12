"""Reproduit le piege documente dans le README/cours : une feature a forte
cardinalite (le montant moyen recent d'un client) peut recevoir une importance
tres elevee dans un RandomForest, alors qu'elle ne porte, par construction,
aucun signal reel sur le risque de retard."""

from collections import defaultdict, deque

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from radar_risque_impaye import generate_invoices, add_recent_late_rate, add_sector_gap
from radar_risque_impaye.features import build_feature_matrix

factures = generate_invoices(n_clients=60, seed=7)
factures = add_sector_gap(add_recent_late_rate(factures, window=5))

# Ajoute manuellement une feature de montant moyen recent (volontairement SANS
# lien avec le risque de retard dans le generateur de donnees).
historique_montant = defaultdict(lambda: deque(maxlen=5))
montant_moyen_recent = []
for row in factures.itertuples():
    hist = historique_montant[row.client]
    montant_moyen_recent.append(float(np.mean(hist)) if hist else np.nan)
    hist.append(row.montant)
factures["montant_moyen_recent"] = montant_moyen_recent
factures = factures.dropna(subset=["montant_moyen_recent"]).reset_index(drop=True)

X, y, groups, cols = build_feature_matrix(
    factures,
    feature_cols=[
        "taux_retard_recent",
        "montant_moyen_recent",
        "ecart_taux_secteur",
        "secteur",
    ],
)

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)

print("=== Avec montant_moyen_recent (feature sans signal reel) ===")
print("accuracy:", round(accuracy_score(y_test, predictions), 4))
for name, imp in sorted(zip(cols, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# --- Sans cette feature ---
X2, y2, groups2, cols2 = build_feature_matrix(
    factures, feature_cols=["taux_retard_recent", "ecart_taux_secteur", "secteur"]
)
train_idx2, test_idx2 = next(splitter.split(X2, y2, groups=groups2))
X2_train, X2_test = X2[train_idx2], X2[test_idx2]
y2_train, y2_test = y2[train_idx2], y2[test_idx2]

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

clf2 = RandomForestClassifier(n_estimators=200, random_state=42)
clf2.fit(X2_train_scaled, y2_train)
predictions2 = clf2.predict(X2_test_scaled)

print("\n=== Sans montant_moyen_recent ===")
print("accuracy:", round(accuracy_score(y2_test, predictions2), 4))
for name, imp in sorted(zip(cols2, clf2.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")
