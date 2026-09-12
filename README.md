# Radar de risque impayé

[![CI](https://github.com/Julionores/radar-risque-impaye/actions/workflows/ci.yml/badge.svg)](https://github.com/Julionores/radar-risque-impaye/actions/workflows/ci.yml)

Un pipeline de **classification supervisée** (scikit-learn) qui prédit si la prochaine facture
d'un client B2B sera payée en retard, à partir de son historique récent — avec, documentés tels
quels, **deux pièges réellement rencontrés** en construisant ce projet : une feature à forte
cardinalité qui reçoit une importance trompeuse, et un modèle « battu » par sa baseline en
accuracy, alors qu'il est le seul des deux réellement utile.

> Projet réalisé par **Junior Tsafack Megnekeu** ([blog.jtmcloud.com](https://blog.jtmcloud.com) ·
> [GitHub](https://github.com/Julionores) ·
> [LinkedIn](https://www.linkedin.com/in/junior-tsafack-megnekeu-b673151b9)) — pièce d'un
> portfolio technique orienté Machine Learning. Voir aussi
> [`gradientforge`](https://github.com/Julionores/gradientforge), un moteur de régression et
> classification codé en NumPy pur,
> [`collecte-agricole-planner`](https://github.com/Julionores/collecte-agricole-planner), et
> l'ensemble du portfolio :
> [`devsecops-pipeline-reference`](https://github.com/Julionores/devsecops-pipeline-reference),
> [`securebank-api`](https://github.com/Julionores/securebank-api),
> [`postgresql-ha-repmgr`](https://github.com/Julionores/postgresql-ha-repmgr),
> [`iso27001-isms-toolkit`](https://github.com/Julionores/iso27001-isms-toolkit),
> [`dynamodb-streams-cdc-pipeline`](https://github.com/Julionores/dynamodb-streams-cdc-pipeline),
> [`aws-troubleshooting-challenge`](https://github.com/Julionores/aws-troubleshooting-challenge),
> [`s3-cross-region-replication`](https://github.com/Julionores/s3-cross-region-replication),
> [`aws-alb-deployment-patterns`](https://github.com/Julionores/aws-alb-deployment-patterns) et
> [`aws-vpc-connectivity-patterns`](https://github.com/Julionores/aws-vpc-connectivity-patterns).
> Ce projet accompagne le module 4 de mon
> [cours Machine Learning & Deep Learning](https://blog.jtmcloud.com/machine-learning/04-classification-supervisee/).

## Le scénario

Une entreprise B2B facture ses clients à crédit. Le service recouvrement veut prioriser ses
relances **avant l'échéance**, à partir de l'historique de facturation (client, secteur
d'activité, montant, délai de paiement observé).

## Pipeline

```
src/radar_risque_impaye/
├── data.py       # Génération d'un historique de facturation B2B synthétique
└── features.py   # Feature engineering temporel sans fuite d'information
```

```python
from radar_risque_impaye import generate_invoices, add_recent_late_rate, add_sector_gap

factures = generate_invoices(n_clients=60, seed=7)
factures = add_sector_gap(add_recent_late_rate(factures, window=5))
```

`add_recent_late_rate` calcule, pour chaque facture, le taux de retard du client sur ses 5
dernières factures — en ne lisant l'historique **qu'avant** de l'y ajouter, pour ne jamais
laisser fuiter d'information du futur dans une feature.

## Piège n°1 — une feature à forte cardinalité peut tromper l'importance d'un RandomForest

```bash
python examples/demo_feature_importance_pitfall.py
```

```
=== Avec montant_moyen_recent (feature sans signal reel) ===
accuracy: 0.6809
  montant_moyen_recent: 0.8275
  secteur_BTP: 0.0506
  taux_retard_recent: 0.0458
  ecart_taux_secteur: 0.0423
  secteur_Services: 0.0179
  secteur_Commerce: 0.0094
  secteur_Industrie: 0.0066

=== Sans montant_moyen_recent ===
accuracy: 0.7787
  secteur_BTP: 0.3250
  taux_retard_recent: 0.2292
  ecart_taux_secteur: 0.2267
  secteur_Services: 0.1226
  secteur_Commerce: 0.0603
  secteur_Industrie: 0.0362
```

`montant_moyen_recent` ne porte, par construction du générateur de données, **aucun lien** avec
le risque de retard — pourtant elle capte 82 % de l'importance d'un `RandomForestClassifier`.
L'importance basée sur l'impureté (Gini) est connue pour être biaisée en faveur des features
continues à forte cardinalité. Une fois écartée, le classement retrouve du sens : le secteur
BTP (le plus tardif dans les données générées) redevient la feature dominante.

## Piège n°2 — l'accuracy peut activement induire en erreur

```bash
python examples/demo_full_pipeline.py
```

```
clients en train: 48 - clients en test: 12

=== Accuracy (metrique trompeuse ici) ===
modele: 0.7652
baseline: 0.7733

=== Metriques adaptees a un probleme desequilibre (class_weight='balanced') ===
recall modele: 0.625
recall baseline: 0.0
precision modele: 0.3535
f1 modele: 0.4516
f1 baseline: 0.0
matrice de confusion:
 [[127  64]
 [ 21  35]]

=== Inference sur deux cas types ===
client BTP a risque: 0.656
client Services fiable: 0.227
```

Sur la seule accuracy, le modèle (76,5 %) fait **moins bien** que la baseline naïve « toujours à
temps » (77,3 %) — parce que seules 23 % des factures sont réellement en retard, la baseline
gagne mécaniquement sans jamais rien détecter. Le `recall` révèle la vraie histoire : la baseline
détecte 0 % des retards (recall = 0), le modèle en détecte 62,5 %. Pour un outil de priorisation
de relances, c'est le modèle qui a de la valeur, malgré son accuracy inférieure.

Le split train/test est réalisé **par client** (`GroupShuffleSplit`), jamais par ligne, pour
qu'aucun client ne soit à la fois dans le train et dans le test — voir
`tests/test_model.py::test_group_split_never_mixes_a_client_across_train_and_test`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

10 tests couvrent : la génération et la reproductibilité des données, l'absence de fuite
d'information dans le feature engineering temporel (vérifiée ligne par ligne), la construction
de la matrice de features, l'étanchéité du split par client, et les deux pièges ci-dessus comme
tests de non-régression.

```
============================= 10 passed in 4.83s ==============================
```

## Installation

```bash
conda create -n radar-risque-impaye python=3.11
conda activate radar-risque-impaye
pip install -r requirements-dev.txt
```

## Structure du projet

```
src/radar_risque_impaye/  # generate_invoices, add_recent_late_rate, add_sector_gap
examples/                 # Les deux pieges, reproductibles
tests/                    # Suite de tests pytest
```

## Licence

MIT — voir [`LICENSE`](LICENSE). Projet à but pédagogique et de démonstration.
