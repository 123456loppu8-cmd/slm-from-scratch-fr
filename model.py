"""Architecture compacte GPT decoder-only, implémentée avec torch.nn."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Attention multi-têtes qui masque les tokens futurs."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model doit être divisible par n_heads.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        # Le triangle inférieur autorise l'accès au passé et au token courant,
        # mais interdit à chaque position d'accéder aux positions futures.
        masque_causal = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer(
            "masque_causal",
            masque_causal.view(1, 1, max_seq_len, max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, longueur, d_model = x.shape

        # Calcul conjoint des requêtes Q, clés K et valeurs V, puis séparation
        # en têtes indépendantes : (lot, têtes, séquence, dimension_par_tête).
        qkv = self.qkv(x)
        qkv = qkv.view(
            batch_size, longueur, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        requetes, cles, valeurs = qkv.unbind(dim=0)

        # Les scores Q·K mesurent la pertinence des positions. La division par
        # sqrt(dimension_par_tête) stabilise leur amplitude avant le softmax.
        scores = (requetes @ cles.transpose(-2, -1)) / (self.head_dim ** 0.5)
        masque = self.masque_causal[:, :, :longueur, :longueur]
        scores = scores.masked_fill(masque == 0, float("-inf"))

        # Le softmax donne les poids de combinaison des vecteurs de valeurs V.
        poids = F.softmax(scores, dim=-1)
        poids = self.attention_dropout(poids)
        sortie = poids @ valeurs
        sortie = sortie.transpose(1, 2).contiguous().view(
            batch_size, longueur, d_model
        )
        return self.residual_dropout(self.proj(sortie))


class FeedForward(nn.Module):
    """Réseau non linéaire appliqué indépendamment à chaque position."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        # Expansion, non-linéarité GELU, puis projection vers la dimension initiale.
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Bloc Transformer avec LayerNorm, sous-couches et connexions résiduelles."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.norm_feed_forward = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Les résidus transmettent l'entrée de chaque sous-couche et facilitent
        # l'optimisation lorsque plusieurs blocs sont empilés.
        x = x + self.attention(self.norm_attention(x))
        x = x + self.feed_forward(self.norm_feed_forward(x))
        return x


class GPTLanguageModel(nn.Module):
    """Petit modèle GPT qui prédit le prochain token."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size doit être positif.")
        if n_layers <= 0 or n_heads <= 0:
            raise ValueError("n_layers et n_heads doivent être positifs.")
        if d_model % n_heads != 0:
            raise ValueError("d_model doit être divisible par n_heads.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len doit être positif.")

        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
        }

        # L'embedding de position indique l'ordre des tokens : l'attention
        # seule ne contient pas cette information séquentielle.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, max_seq_len, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.language_model_head = nn.Linear(d_model, vocab_size, bias=False)

        # Le partage des poids lie la représentation des tokens à leur prédiction.
        self.language_model_head.weight = self.token_embedding.weight
        self.apply(self._initialiser_poids)

    @staticmethod
    def _initialiser_poids(module: nn.Module):
        """Initialisation normale des poids et nulle des biais."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets=None):
        """Retourne les logits et, si des cibles sont fournies, la Cross-Entropy."""
        _, longueur = input_ids.shape
        if longueur > self.config["max_seq_len"]:
            raise ValueError(
                f"Séquence trop longue ({longueur}); maximum "
                f"{self.config['max_seq_len']}."
            )

        positions = torch.arange(longueur, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for bloc in self.blocks:
            x = bloc(x)

        logits = self.language_model_head(self.final_norm(x))
        perte = None
        if targets is not None:
            # Aplatissement des positions pour comparer chaque distribution
            # prédite à l'indice du prochain token attendu.
            perte = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, perte
