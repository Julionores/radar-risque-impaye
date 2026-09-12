import numpy as np
import pandas as pd

from radar_risque_impaye import add_recent_late_rate, add_sector_gap
from radar_risque_impaye.features import build_feature_matrix


def _toy_invoices():
    return pd.DataFrame(
        {
            "client": ["A", "A", "A", "B", "B"],
            "secteur": ["BTP", "BTP", "BTP", "Services", "Services"],
            "date_facture": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-15", "2024-02-15"]
            ),
            "montant": [100.0, 100.0, 100.0, 200.0, 200.0],
            "delai_paiement_jours": [10, 40, 40, 10, 10],
            "paye_en_retard": [0, 1, 1, 0, 0],
        }
    )


def test_first_invoice_per_client_has_no_history_and_is_dropped():
    factures = _toy_invoices()
    result = add_recent_late_rate(factures, window=5)
    # 2 clients -> 2 premieres factures sans historique, donc ecartees
    assert len(result) == len(factures) - 2


def test_recent_late_rate_never_uses_the_current_or_future_row():
    factures = _toy_invoices()
    result = add_recent_late_rate(factures, window=5)

    # 2eme facture du client A : ne doit voir QUE la 1ere facture (paye_en_retard=0)
    row_a2 = result[(result["client"] == "A")].iloc[0]
    assert row_a2["taux_retard_recent"] == 0.0

    # 3eme facture du client A : doit voir les factures 1 et 2 (0 et 1) -> moyenne 0.5
    row_a3 = result[(result["client"] == "A")].iloc[1]
    assert row_a3["taux_retard_recent"] == 0.5


def test_sector_gap_averages_to_zero_within_each_sector():
    """ecart_taux_secteur est defini comme une deviation par rapport a la
    moyenne de son propre secteur : par construction, sa moyenne au sein
    d'un meme secteur doit toujours etre nulle, quelles que soient les
    donnees."""
    factures = _toy_invoices()
    with_history = add_recent_late_rate(factures, window=5)
    with_gap = add_sector_gap(with_history)

    moyennes_par_secteur = with_gap.groupby("secteur")["ecart_taux_secteur"].mean()
    assert np.allclose(moyennes_par_secteur, 0.0)


def test_build_feature_matrix_shapes():
    factures = _toy_invoices()
    with_history = add_sector_gap(add_recent_late_rate(factures, window=5))
    X, y, groups, cols = build_feature_matrix(
        with_history,
        feature_cols=["taux_retard_recent", "ecart_taux_secteur", "secteur"],
    )
    assert X.shape[0] == len(with_history)
    assert X.shape[1] == len(cols)
    assert set(groups) == {"A", "B"}
    assert "secteur_BTP" in cols and "secteur_Services" in cols
