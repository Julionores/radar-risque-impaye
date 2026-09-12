import numpy as np
import pytest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from radar_risque_impaye import generate_invoices, add_recent_late_rate, add_sector_gap
from radar_risque_impaye.features import build_feature_matrix


@pytest.fixture(scope="module")
def split_dataset():
    factures = generate_invoices(n_clients=60, seed=7)
    factures = add_sector_gap(add_recent_late_rate(factures, window=5))
    X, y, groups, cols = build_feature_matrix(
        factures, feature_cols=["taux_retard_recent", "ecart_taux_secteur", "secteur"]
    )

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X, y, groups, cols, train_idx, test_idx


def test_group_split_never_mixes_a_client_across_train_and_test(split_dataset):
    _, _, groups, _, train_idx, test_idx = split_dataset
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_default_random_forest_matches_documented_pitfall(split_dataset):
    """Non-regression : sur ce jeu de donnees, meme sans feature de montant,
    l'accuracy brute d'un RandomForest par defaut ne bat PAS necessairement une
    baseline naive -- c'est le point pedagogique du module 4 du cours."""
    X, y, _, _, train_idx, test_idx = split_dataset
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_scaled, y_train)
    predictions = clf.predict(X_test_scaled)

    baseline_pred = np.zeros_like(y_test)
    model_accuracy = accuracy_score(y_test, predictions)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    # On ne force pas le modele a battre la baseline en accuracy : on verifie
    # seulement que le resultat documente dans le cours reste stable.
    assert model_accuracy == pytest.approx(0.7652, abs=1e-4)
    assert baseline_accuracy == pytest.approx(0.7733, abs=1e-4)


def test_balanced_model_beats_baseline_on_recall_and_f1(split_dataset):
    """C'est la metrique adaptee (recall/F1), pas l'accuracy, qui doit montrer
    la valeur reelle du modele sur un probleme desequilibre."""
    X, y, _, _, train_idx, test_idx = split_dataset
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train_scaled, y_train)
    predictions = clf.predict(X_test_scaled)
    baseline_pred = np.zeros_like(y_test)

    assert recall_score(y_test, predictions) > recall_score(y_test, baseline_pred)
    assert f1_score(y_test, predictions) > f1_score(y_test, baseline_pred)
    assert precision_score(y_test, predictions) > 0
