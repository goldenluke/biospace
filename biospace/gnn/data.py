"""
biospace.gnn.data
====================

Ponte entre `biospace.graph.build_cohort_similarity_graph` (a rede de
pacientes) + `RepresentationSpace` (as Features) + rótulos (ex.:
fenótipo) e os arrays (A, X, y, labeled_mask) que `SimpleGCN` consome.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["prepare_node_classification_data"]


def prepare_node_classification_data(
    space: "RepresentationSpace",
    graph: nx.Graph,
    labels: dict[str, str],
    labeled_ids: Sequence[str],
    order: Optional[Sequence[str]] = None,
) -> dict:
    """
    `labels`: rótulo (ex.: nome do fenótipo) por system_id — usado como
    alvo. `labeled_ids`: quais desses IDs entram como TREINO (rotulados)
    — os demais nós do grafo participam da propagação de mensagens, mas
    não da perda (o experimento semi-supervisionado e transdutivo
    original de Kipf & Welling, 2017).

    Retorna um dict com `A` (adjacência, ordem = `node_ids`), `X`
    (Features), `y` (rótulo inteiro, 0 onde desconhecido — excluído da
    perda via `labeled_mask`), `labeled_mask`, `node_ids`, `label_names`
    (lista ordenada de nomes de rótulo, `y[i]` é o índice em `label_names`).
    """
    node_ids = list(graph.nodes())
    used_order = order or space.order()

    X = np.stack([space.get(sid).as_vector(used_order) for sid in node_ids])

    label_names = sorted(set(labels.values()))
    label_index = {name: i for i, name in enumerate(label_names)}

    y_raw = np.array([label_index.get(labels.get(sid), -1) for sid in node_ids])
    labeled_ids_set = set(labeled_ids)
    labeled_mask = np.array([sid in labeled_ids_set and sid in labels for sid in node_ids])

    A = nx.to_numpy_array(graph, nodelist=node_ids, weight=None)

    return {
        "A": A,
        "X": X,
        "y": np.where(y_raw < 0, 0, y_raw),
        "labeled_mask": labeled_mask,
        "node_ids": node_ids,
        "label_names": label_names,
    }
