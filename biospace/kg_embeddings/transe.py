"""
biospace.kg_embeddings.transe
=================================

TransE (Bordes et al., 2013) — o algoritmo clássico de embedding de
grafo de conhecimento: cada entidade e cada relação recebem um vetor
no mesmo espaço, e uma tripla (h, r, t) é "verdadeira" na medida em
que h + r ≈ t (a relação é literalmente um vetor de translação entre
a entidade cabeça e a entidade cauda). Implementado em NumPy puro —
forward, gradiente e negative sampling escritos à mão, sem framework
de deep learning disponível no ambiente (mesma situação já enfrentada
em `biospace.sequence`, mesma solução: implementar o método de
verdade, não fingir com um substituto mais simples).

Validado contra um grafo sintético com estrutura CONHECIDA (predição
de aresta faltante com verdade conhecida) antes de qualquer aplicação
real — a avaliação clássica e decisiva de embeddings de grafo de
conhecimento, não apenas "o treino roda sem erro".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["TransE", "Triple"]


@dataclass
class Triple:
    head: str
    relation: str
    tail: str


@dataclass
class TransE:
    dim: int = 16
    margin: float = 1.0
    lr: float = 0.05
    n_epochs: int = 200
    seed: int = 0
    entity_emb: Optional[dict] = field(default=None, repr=False)
    relation_emb: Optional[dict] = field(default=None, repr=False)

    def fit(self, triples: list[Triple], all_entities: Optional[list[str]] = None) -> list[float]:
        """`all_entities`: conjunto completo de entidades a embutir, incluindo as que só aparecem
        em triplas retidas para avaliação (nunca vistas no treino) -- protocolo padrão de
        avaliação transdutiva de embeddings de grafo: todas as entidades são conhecidas de
        antemão, só algumas TRIPLAS (fatos) são retidas para testar predição de link. Sem isso,
        uma entidade que só aparece em triplas de teste nunca ganharia embedding."""
        rng = np.random.default_rng(self.seed)
        entidades = sorted(({t.head for t in triples} | {t.tail for t in triples}) | set(all_entities or []))
        relacoes = sorted({t.relation for t in triples})

        limite = 6.0 / np.sqrt(self.dim)
        self.entity_emb = {e: rng.uniform(-limite, limite, self.dim) for e in entidades}
        self.relation_emb = {r: rng.uniform(-limite, limite, self.dim) for r in relacoes}
        for e in entidades:
            self.entity_emb[e] /= np.linalg.norm(self.entity_emb[e])

        historico_perda = []
        entidades_arr = np.array(entidades, dtype=object)

        for epoca in range(self.n_epochs):
            perda_epoca = 0.0
            ordem = rng.permutation(len(triples))
            for idx in ordem:
                t = triples[idx]
                if rng.random() < 0.5:
                    t_neg = Triple(str(rng.choice(entidades_arr)), t.relation, t.tail)
                else:
                    t_neg = Triple(t.head, t.relation, str(rng.choice(entidades_arr)))

                h, r, tail = self.entity_emb[t.head], self.relation_emb[t.relation], self.entity_emb[t.tail]
                h_n, tail_n = self.entity_emb[t_neg.head], self.entity_emb[t_neg.tail]

                score_pos = np.linalg.norm(h + r - tail)
                score_neg = np.linalg.norm(h_n + r - tail_n)
                perda = max(0.0, self.margin + score_pos - score_neg)
                perda_epoca += perda

                if perda > 0:
                    diff_pos = h + r - tail
                    diff_neg = h_n + r - tail_n
                    grad_pos = diff_pos / (np.linalg.norm(diff_pos) + 1e-9)
                    grad_neg = diff_neg / (np.linalg.norm(diff_neg) + 1e-9)

                    self.entity_emb[t.head] -= self.lr * grad_pos
                    self.entity_emb[t.tail] += self.lr * grad_pos
                    self.relation_emb[t.relation] -= self.lr * grad_pos

                    self.entity_emb[t_neg.head] += self.lr * grad_neg
                    self.entity_emb[t_neg.tail] -= self.lr * grad_neg
                    self.relation_emb[t.relation] += self.lr * grad_neg

                    for e in {t.head, t.tail, t_neg.head, t_neg.tail}:
                        norma = np.linalg.norm(self.entity_emb[e])
                        if norma > 1e-9:
                            self.entity_emb[e] /= norma

            historico_perda.append(perda_epoca / len(triples))
        return historico_perda

    def score(self, head: str, relation: str, tail: str) -> float:
        """Menor = mais provavel que a tripla seja verdadeira (distancia h+r-t)."""
        h, r, t = self.entity_emb[head], self.relation_emb[relation], self.entity_emb[tail]
        return float(np.linalg.norm(h + r - t))

    def predict_tail(self, head: str, relation: str, candidates: list[str], k: int = 1) -> list[str]:
        """Rankeia `candidates` por plausibilidade como cauda de (head, relation, ?) -- os k mais plausiveis primeiro."""
        scores = [(c, self.score(head, relation, c)) for c in candidates]
        scores.sort(key=lambda x: x[1])
        return [c for c, _ in scores[:k]]
