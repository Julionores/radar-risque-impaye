"""Feature engineering temporel sans fuite d'information.

Regle stricte : la feature d'une facture ne doit jamais utiliser d'information
posterieure a cette facture (ni la facture elle-meme, ni celles qui suivent).
"""

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def add_recent_late_rate(factures, window=5):
    """Ajoute `taux_retard_recent` : taux de retard du client sur ses `window`
    dernieres factures, calcule AVANT que la facture courante ne soit ajoutee
    a son propre historique."""
    historique = defaultdict(lambda: deque(maxlen=window))
    taux_retard_recent = []

    for row in factures.itertuples():
        hist = historique[row.client]
        if len(hist) == 0:
            taux_retard_recent.append(np.nan)
        else:
            taux_retard_recent.append(float(np.mean([h for h in hist])))
        historique[row.client].append(row.paye_en_retard)

    factures = factures.copy()
    factures["taux_retard_recent"] = taux_retard_recent
    return factures.dropna(subset=["taux_retard_recent"]).reset_index(drop=True)


def add_sector_gap(factures):
    """Ajoute `ecart_taux_secteur` : ecart entre le taux de retard recent
    d'une facture et la moyenne observee dans son secteur."""
    factures = factures.copy()
    moyenne_secteur = factures.groupby("secteur")["taux_retard_recent"].transform(
        "mean"
    )
    factures["ecart_taux_secteur"] = factures["taux_retard_recent"] - moyenne_secteur
    return factures


def build_feature_matrix(factures, feature_cols):
    """Encode le secteur en one-hot et retourne (X, y, groups, colonnes_utilisees)."""
    encoded = pd.get_dummies(factures, columns=["secteur"], prefix="secteur")
    secteur_cols = [c for c in encoded.columns if c.startswith("secteur_")]
    all_cols = [c for c in feature_cols if c != "secteur"] + secteur_cols
    X = encoded[all_cols].values
    y = encoded["paye_en_retard"].values
    groups = encoded["client"].values
    return X, y, groups, all_cols
