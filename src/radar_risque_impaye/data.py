"""Generation d'un historique de facturation B2B synthetique.

Le taux de retard depend du secteur d'activite du client (le BTP paie
structurellement plus en retard que les autres secteurs) : c'est ce signal
que le modele du module `features`/`examples` doit retrouver.
"""

import numpy as np
import pandas as pd

SECTEURS = ["Industrie", "Commerce", "Services", "BTP"]
TAUX_RETARD_SECTEUR = {
    "Industrie": 0.15,
    "Commerce": 0.20,
    "Services": 0.10,
    "BTP": 0.40,
}


def generate_invoices(n_clients=60, seed=7):
    """Genere un DataFrame de factures pour n_clients clients repartis sur 4 secteurs."""
    rng = np.random.default_rng(seed)
    client_secteur = {f"client_{i:03d}": rng.choice(SECTEURS) for i in range(n_clients)}

    rows = []
    date_debut = pd.Timestamp("2024-01-01")
    for client, secteur in client_secteur.items():
        n_factures = rng.integers(15, 30)
        dates = sorted(
            date_debut + pd.to_timedelta(rng.integers(0, 700, n_factures), unit="D")
        )
        montant_base = rng.uniform(500, 8000)
        for date in dates:
            montant = montant_base * rng.uniform(0.7, 1.3)
            en_retard = rng.random() < TAUX_RETARD_SECTEUR[secteur]
            delai = rng.integers(35, 90) if en_retard else rng.integers(5, 30)
            rows.append(
                {
                    "client": client,
                    "secteur": secteur,
                    "date_facture": date,
                    "montant": round(float(montant), 2),
                    "delai_paiement_jours": int(delai),
                    "paye_en_retard": int(en_retard),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["client", "date_facture"])
        .reset_index(drop=True)
    )
