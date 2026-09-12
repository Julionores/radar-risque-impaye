from radar_risque_impaye import generate_invoices
from radar_risque_impaye.data import TAUX_RETARD_SECTEUR


def test_generate_invoices_shape_and_columns():
    factures = generate_invoices(n_clients=10, seed=1)
    assert len(factures) > 0
    assert set(factures.columns) == {
        "client",
        "secteur",
        "date_facture",
        "montant",
        "delai_paiement_jours",
        "paye_en_retard",
    }
    assert factures["client"].nunique() == 10


def test_generate_invoices_is_reproducible():
    a = generate_invoices(n_clients=20, seed=42)
    b = generate_invoices(n_clients=20, seed=42)
    assert a.equals(b)


def test_btp_has_a_higher_late_rate_than_services():
    factures = generate_invoices(n_clients=200, seed=7)
    taux_par_secteur = factures.groupby("secteur")["paye_en_retard"].mean()
    assert taux_par_secteur["BTP"] > taux_par_secteur["Services"]
    # coherent avec la configuration du generateur (a 200 clients, l'ecart est net)
    assert taux_par_secteur["BTP"] > TAUX_RETARD_SECTEUR["Services"]
