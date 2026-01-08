# OptiFolio

**Version 1.0**

OptiFolio est un système d'optimisation de portefeuille basé sur l'apprentissage par renforcement (Reinforcement Learning) utilisant l'algorithme PPO (Proximal Policy Optimization). Le projet permet d'entraîner un agent intelligent à gérer un portefeuille d'actifs financiers en optimisant les rendements tout en contrôlant la volatilité cible.

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Pipeline de données](#-pipeline-de-données)
- [Environnement d'entraînement](#-environnement-dentraînement)
- [Technologies utilisées](#-technologies-utilisées)
- [Auteur](#-auteur)

## ✨ Fonctionnalités

- **Téléchargement automatique** de données de marché depuis Yahoo Finance
- **Nettoyage et préparation** des données financières (prix, volumes, rendements)
- **Extraction de features** avancées :
  - Volatilité EWMA (court et long terme)
  - Corrélations moyennes entre actifs
  - Rendements normalisés
- **Environnement d'apprentissage par renforcement** personnalisé (Gymnasium)
- **Entraînement PPO** avec contrôle de la volatilité cible
- **Visualisation** des données et des features
- **Suivi des performances** via TensorBoard

## 🏗️ Architecture

Le projet suit une architecture modulaire organisée en plusieurs composants :

1. **Gestion des données** (`utils/dataHandler.py`) : Téléchargement et nettoyage des données brutes
2. **Extraction de features** (`utils/featuresHandler.py`) : Calcul de métriques financières avancées
3. **Création du dataset** (`utils/datasetHandler.py`) : Agrégation des features en un dataset unifié
4. **Environnement RL** (`env/optiFolioEnv.py`) : Environnement Gymnasium personnalisé pour l'entraînement
5. **Entraînement** (`main.ipynb`) : Script principal d'entraînement de l'agent PPO

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip ou conda

### Installation des dépendances

```bash
# Cloner le repository (si applicable)
git clone <repository-url>
cd OptiFolio

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

- `yfinance` : Téléchargement de données financières
- `pandas` : Manipulation de données
- `numpy` : Calculs numériques
- `gymnasium` : Framework d'environnements RL
- `stable_baselines3` : Implémentation PPO
- `tensorboard` : Visualisation des métriques d'entraînement
- `matplotlib` : Visualisation de données
- `jupyterlab` : Environnement de développement interactif

## 📖 Utilisation

### 1. Préparation des données

#### Étape 1 : Téléchargement et nettoyage des données brutes

```python
from utils.dataHandler import main, tickers_list

# Télécharge les données de marché et génère les fichiers nettoyés
main(tickers_list)
```

Les indices boursiers téléchargés par défaut sont :
- `^NDX` : NASDAQ-100
- `^FCHI` : CAC 40
- `^GDAXI` : DAX
- `^N225` : Nikkei 225
- `^HSI` : Hang Seng
- `^SSMI` : SMI Suisse

#### Étape 2 : Extraction des features

```python
from utils.featuresHandler import *

# Calcule les features financières (volatilité, corrélations, etc.)
# Le script s'exécute automatiquement si appelé directement
```

#### Étape 3 : Création du dataset final

```python
from utils.datasetHandler import create_dataset

# Combine toutes les features en un dataset synchronisé
create_dataset("data/features")
```

### 2. Entraînement de l'agent

Ouvrez `main.ipynb` dans JupyterLab et exécutez les cellules pour :

1. Charger le dataset préparé
2. Initialiser l'environnement `optiFolioEnv`
3. Entraîner l'agent PPO
4. Visualiser les résultats via TensorBoard

### 3. Visualisation

Les notebooks dans le dossier `notebook/` permettent de visualiser :
- Les données brutes et nettoyées (`dataVisualization.ipynb`)
- Les features extraites (`featureVisualization.ipynb`)

## 📁 Structure du projet

```
OptiFolio/
├── data/
│   ├── raw/              # Données brutes téléchargées
│   │   └── yahoo/
│   ├── cleaned/          # Données nettoyées (prix, volumes, rendements)
│   ├── features/         # Features calculées (volatilité, corrélations)
│   └── dataset/          # Dataset final combiné
├── env/
│   └── optiFolioEnv.py   # Environnement Gymnasium personnalisé
├── utils/
│   ├── dataHandler.py    # Gestion des données brutes
│   ├── featuresHandler.py # Extraction de features
│   └── datasetHandler.py  # Création du dataset final
├── notebook/
│   ├── dataVisualization.ipynb
│   └── featureVisualization.ipynb
├── ppo_tensorboard/      # Logs TensorBoard de l'entraînement
├── main.ipynb            # Script principal d'entraînement
├── requirements.txt      # Dépendances Python
└── README.md             # Documentation
```

## 🔄 Pipeline de données

Le pipeline de traitement des données suit ces étapes :

1. **Téléchargement** : Récupération des données historiques depuis Yahoo Finance
2. **Nettoyage** : Extraction des prix de clôture, volumes et calcul des rendements
3. **Feature Engineering** :
   - Volatilité EWMA (λ court = 0.94, λ long = 0.97)
   - Ratio de volatilité (court/long terme)
   - Corrélations moyennes EWMA
   - Rendements normalisés par volatilité
4. **Agrégation** : Synchronisation de toutes les features sur des dates communes
5. **Dataset final** : Création d'un fichier CSV unifié pour l'entraînement

## 🎮 Environnement d'entraînement

L'environnement `optiFolioEnv` est configuré avec les paramètres suivants :

- **Capital initial** : 10 000 (par défaut)
- **Fenêtre de lookback** : 20 jours
- **Durée maximale** : 252 jours (1 année de trading)
- **Volatilité cible** : 2% (par défaut)

### Fonction de récompense

La récompense combine plusieurs composantes :

- **Rendement logarithmique** : Récompense basée sur le rendement du portefeuille
- **Bonus alpha** : Bonus pour les trades performants
- **Pénalité de volatilité** : Pénalité si la volatilité dépasse la cible
- **Pénalité de turnover** : Décourage les réallocations excessives

### Espace d'observation

- Fenêtre glissante des features sur les N derniers jours
- Volatilité cible comme feature supplémentaire

### Espace d'action

- Poids de portefeuille pour chaque actif (normalisés via softmax)

## 🛠️ Technologies utilisées

- **Python** : Langage de programmation principal
- **Yahoo Finance API** : Source de données de marché
- **Gymnasium** : Standard pour les environnements RL
- **Stable-Baselines3** : Bibliothèque d'algorithmes RL
- **TensorBoard** : Outil de visualisation des métriques
- **Pandas/NumPy** : Manipulation et calculs sur les données

## 📊 Métriques et suivi

Les métriques d'entraînement sont enregistrées dans `ppo_tensorboard/` et peuvent être visualisées avec :

```bash
tensorboard --logdir=ppo_tensorboard
```

## 🔧 Configuration avancée

### Personnalisation des indices

Modifiez la liste `tickers_list` dans `utils/dataHandler.py` pour ajouter ou retirer des indices.

### Paramètres de l'environnement

Les paramètres de l'environnement peuvent être ajustés lors de l'initialisation :

```python
env = optiFolioEnv(
    dataset_path="data/dataset/dataset.csv",
    initial_amount=10_000,
    lookback=20,
    max_days=252,
    target_vol=0.02
)
```

### Paramètres EWMA

Les facteurs de décroissance pour les calculs EWMA peuvent être modifiés dans `utils/featuresHandler.py` :
- `lambda_short = 0.94` (court terme)
- `lambda_long = 0.97` (long terme)

## ⚠️ Avertissements

- Ce projet est à des fins éducatives et de recherche
- Les performances passées ne garantissent pas les résultats futurs
- Toujours effectuer des tests approfondis avant toute utilisation en production
- Les données de marché peuvent contenir des erreurs ou des lacunes

## 📝 Notes de version

### Version 1.0

- Implémentation initiale du pipeline de données
- Environnement RL personnalisé avec PPO
- Extraction de features financières avancées
- Support de multiples indices boursiers
- Visualisation via TensorBoard

## 👤 Auteur

Développé dans le cadre d'un projet d'optimisation de portefeuille.

---

**License** : Ce projet est fourni tel quel, sans garantie d'aucune sorte.
