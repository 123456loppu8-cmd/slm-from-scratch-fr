"""Entraînement local d'un GPT compact, puis génération d'un exemple."""

import argparse
import logging
import random
from pathlib import Path

import torch

from model import GPTLanguageModel
from tokenizer import BOS_TOKEN, EOS_TOKEN, load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")


def choisir_appareil() -> torch.device:
    """Choisit CUDA, puis MPS, puis CPU selon les capacités disponibles."""
    if torch.cuda.is_available():
        appareil = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        appareil = torch.device("mps")
    else:
        appareil = torch.device("cpu")
    logger.info("Appareil utilisé : %s", appareil)
    return appareil


def creer_batch(tokens, batch_size: int, seq_len: int, device: torch.device):
    """Crée des entrées x et des cibles y décalées d'un token."""
    max_debut = len(tokens) - seq_len
    if max_debut <= 0:
        raise ValueError("Pas assez de tokens pour créer un batch.")
    debuts = torch.randint(0, max_debut, (batch_size,))
    x = torch.stack([tokens[i : i + seq_len] for i in debuts])
    y = torch.stack([tokens[i + 1 : i + seq_len + 1] for i in debuts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluer_perte(
    model: GPTLanguageModel,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    nombre_batches: int = 10,
):
    """Estime la perte moyenne sur les données d'entraînement et de validation."""
    model.eval()
    resultats = {}
    for nom, donnees in (("train", train_tokens), ("val", val_tokens)):
        longueur_locale = min(seq_len, len(donnees) - 1)
        if longueur_locale < 1:
            resultats[nom] = float("nan")
            continue
        pertes = []
        for _ in range(nombre_batches):
            x, y = creer_batch(donnees, batch_size, longueur_locale, device)
            _, perte = model(x, y)
            pertes.append(perte.item())
        resultats[nom] = sum(pertes) / len(pertes)
    model.train()
    return resultats


@torch.no_grad()
def generer(
    model: GPTLanguageModel,
    tokenizer,
    prompt: str,
    device: torch.device,
    nombre_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    """Génère les tokens un à un en ne regardant que le contexte précédent."""
    if temperature <= 0:
        raise ValueError("temperature doit être strictement positive.")

    model.eval()
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if bos_id is None:
        raise RuntimeError("Le tokenizer ne contient pas le token <bos>.")

    # Le prompt n'est pas encadré par BOS/EOS automatiquement : on ajoute BOS
    # explicitement, puis la génération peut continuer au-delà du texte fourni.
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long, device=device)

    for _ in range(nombre_tokens):
        contexte = input_ids[:, -model.config["max_seq_len"] :]
        logits, _ = model(contexte)
        logits = logits[:, -1, :] / temperature

        # Top-k limite l'échantillonnage aux k tokens ayant les scores les plus élevés.
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            seuil, _ = torch.topk(logits, k)
            logits[logits < seuil[:, [-1]]] = float("-inf")

        probabilites = torch.softmax(logits, dim=-1)
        prochain_token = torch.multinomial(probabilites, num_samples=1)
        input_ids = torch.cat((input_ids, prochain_token), dim=1)
        if eos_id is not None and prochain_token.item() == eos_id:
            break

    model.train()
    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle GPT depuis un corpus local."
    )
    parser.add_argument("--data", default="data.txt")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--output", default="mon_modele.pt")
    parser.add_argument("--prompt", default="Bonjour")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--generate-tokens", type=int, default=100)
    args = parser.parse_args()

    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("--steps et --batch-size doivent être positifs.")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval doit être positif.")
    if args.d_model % args.n_heads != 0:
        raise ValueError("--d-model doit être divisible par --n-heads.")

    random.seed(42)
    torch.manual_seed(42)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier absent : {data_path}.")
    texte = data_path.read_text(encoding="utf-8").strip()
    if not texte:
        raise ValueError(f"Le fichier {data_path} est vide.")

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    token_ids = tokenizer.encode(texte).ids
    if len(token_ids) < 10:
        raise ValueError("Le corpus est trop petit; ajoutez du texte à data.txt.")

    tous_tokens = torch.tensor(token_ids, dtype=torch.long)
    index_separation = int(0.9 * len(tous_tokens))
    index_separation = max(2, min(index_separation, len(tous_tokens) - 2))
    train_tokens = tous_tokens[:index_separation]
    val_tokens = tous_tokens[index_separation:]

    device = choisir_appareil()
    seq_len = min(args.seq_len, len(train_tokens) - 1)
    if seq_len < 1:
        raise ValueError("Corpus insuffisant pour entraîner le modèle.")

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=seq_len,
        dropout=args.dropout,
    ).to(device)

    nombre_parametres = sum(param.numel() for param in model.parameters())
    logger.info("Vocabulaire : %s tokens", vocab_size)
    logger.info("Tokens du corpus : %s", len(tous_tokens))
    logger.info("Paramètres du modèle : %.2f millions", nombre_parametres / 1e6)
    logger.info("Longueur de séquence : %s", seq_len)

    optimiseur = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for etape in range(1, args.steps + 1):
        x, y = creer_batch(train_tokens, args.batch_size, seq_len, device)
        _, perte = model(x, y)

        # Rétropropagation et mise à jour des poids par AdamW.
        optimiseur.zero_grad(set_to_none=True)
        perte.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiseur.step()

        if etape == 1 or etape % args.eval_interval == 0 or etape == args.steps:
            pertes = evaluer_perte(
                model, train_tokens, val_tokens, args.batch_size, seq_len, device
            )
            logger.info(
                "Étape %s/%s | perte entraînement : %.4f | perte validation : %.4f",
                etape,
                args.steps,
                pertes["train"],
                pertes["val"],
            )

    sortie = Path(args.output)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
            "tokenizer_path": str(Path(args.tokenizer)),
        },
        sortie,
    )
    logger.info("Poids et configuration enregistrés dans %s", sortie)

    exemple = generer(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        nombre_tokens=args.generate_tokens,
    )
    logger.info("Exemple généré :\n%s", exemple)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Échec de l'entraînement ou de la génération.")
        raise
