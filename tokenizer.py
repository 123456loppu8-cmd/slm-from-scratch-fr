"""Entraînement et chargement d'un tokenizer BPE depuis un fichier texte local."""

import argparse
import logging
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("tokenizer")

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def train_tokenizer(
    data_path: str = "data.txt",
    output_path: str = "tokenizer.json",
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Entraîne un tokenizer BPE sur le texte du fichier indiqué."""
    source = Path(data_path)
    destination = Path(output_path)

    if not source.exists():
        raise FileNotFoundError(
            f"Fichier absent : {source}. Créez-le avant d'entraîner le tokenizer."
        )
    if source.stat().st_size == 0:
        raise ValueError(f"Le fichier {source} est vide.")
    if vocab_size < 10:
        raise ValueError("vocab_size doit être au moins égal à 10.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    # BPE construit progressivement un vocabulaire de sous-mots.
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))

    # ByteLevel permet de représenter les caractères sous forme d'octets,
    # y compris les caractères accentués ou rares.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[UNK_TOKEN, BOS_TOKEN, EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    logger.info("Entraînement BPE sur %s", source)
    tokenizer.train(files=[str(source)], trainer=trainer)

    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise RuntimeError("Impossible de trouver les tokens BOS/EOS.")

    # Encadre automatiquement chaque texte encodé avec BOS et EOS.
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",
        special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
    )

    tokenizer.save(str(destination))
    logger.info(
        "Tokenizer enregistré : %s (vocabulaire de %s tokens)",
        destination,
        tokenizer.get_vocab_size(),
    )
    return tokenizer


def load_tokenizer(path: str = "tokenizer.json") -> Tokenizer:
    """Charge un tokenizer sauvegardé."""
    tokenizer_path = Path(path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer absent : {tokenizer_path}. Lancez d'abord tokenizer.py."
        )
    return Tokenizer.from_file(str(tokenizer_path))


def main():
    parser = argparse.ArgumentParser(description="Entraîne un tokenizer BPE local.")
    parser.add_argument("--data", default="data.txt", help="Fichier texte du corpus.")
    parser.add_argument("--output", default="tokenizer.json", help="Fichier de sortie.")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    tokenizer = train_tokenizer(
        data_path=args.data,
        output_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    exemple = "Bonjour, ceci est un test."
    encodage = tokenizer.encode(exemple)
    logger.info("Exemple encodé : %s", encodage.ids)
    logger.info("Exemple décodé : %s", tokenizer.decode(encodage.ids))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Échec de l'entraînement du tokenizer.")
        raise
