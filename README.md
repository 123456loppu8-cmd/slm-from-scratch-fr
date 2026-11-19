# SLM from scratch — projet pédagogique

Petit modèle de langage GPT decoder-only, initialisé aléatoirement et entraîné uniquement sur un fichier texte local. Le tokenizer BPE est également appris sur ce corpus ; aucun modèle pré-entraîné n'est téléchargé.

## Fichiers

- `data.txt` : petit corpus français pour vérifier que le projet fonctionne. Remplacez-le ou enrichissez-le avec un corpus pertinent.
- `tokenizer.py` : entraîne et sauvegarde le tokenizer BPE dans `tokenizer.json`.
- `model.py` : architecture GPT en PyTorch (`torch.nn`).
- `train_and_generate.py` : boucle d'entraînement AdamW/Cross-Entropy, sauvegarde `mon_modele.pt`, puis génère un exemple.
- `requirements.txt` : dépendances Python.

## Installation

Utilisez Python 3.10 ou plus récent. Installez PyTorch avec la commande correspondant à votre système depuis le sélecteur officiel : https://pytorch.org/get-started/locally/ . Puis installez les dépendances :

```bash
python -m venv .venv
# Linux/macOS :
source .venv/bin/activate
# Windows PowerShell : .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Lancer un premier test

Depuis la racine du dépôt :

```bash
python tokenizer.py --data data.txt --vocab-size 8000
python train_and_generate.py --data data.txt --tokenizer tokenizer.json --steps 200 --prompt "Bonjour"
```

Le modèle de démonstration est petit et le corpus inclus est très court : ces commandes vérifient le pipeline, mais ne produiront pas un assistant de qualité. Pour des résultats plus utiles, ajoutez un corpus substantiel, autorisé et représentatif. Le nombre d'étapes, la taille du modèle et la taille des batches peuvent être ajustés selon la mémoire disponible.

Le script détecte automatiquement CUDA, MPS sur Mac, ou bascule sur CPU. Le CPU peut être lent, même pour ce petit modèle.
