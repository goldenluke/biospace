"""
tests.test_core_hardening
============================

Um teste por endurecimento feito na auditoria do núcleo (validação de
entrada, mensagens de erro claras, casos de borda) — cada um explicando
POR QUE a validação existe, não só QUE ela existe.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from biospace.core import (
    Cohort,
    CompositeRepresentation,
    Feature,
    Normal,
    Representation,
    RepresentationSpace,
    SemanticDomain,
    Trajectory,
)
from biospace.plugins.sleep import ApneaDomain, HypoxiaDomain


class _DomainSemName(SemanticDomain):
    """Domínio que 'esquece' de definir `name` — usado só para provar a validação."""

    def __init__(self):
        super().__init__([])

    def encode(self, measurements):
        return [Feature(name="x", value=0.0)]


def test_semantic_domain_requires_name():
    """SemanticDomain sem `name` deve falhar NA CONSTRUÇÃO, não em algum ponto confuso mais tarde."""
    with pytest.raises(ValueError, match="name"):
        _DomainSemName()


def test_representation_rejects_duplicate_domain_names():
    """
    Regressão direta do bug encontrado manualmente ao testar check_extensibility():
    dois domínios com o mesmo `.name` na mesma Representation colidiam
    silenciosamente no dict de componentes.
    """
    with pytest.raises(ValueError, match="colidentes"):
        Representation([ApneaDomain(), ApneaDomain()])


def test_representation_extend_also_rejects_collision():
    """extend() cria uma nova Representation por baixo dos panos — a mesma validação deve se aplicar."""
    representation = Representation([ApneaDomain()])
    with pytest.raises(ValueError, match="colidentes"):
        representation.extend(ApneaDomain())


def test_representation_accepts_distinct_names():
    """Contraprova: domínios com nomes distintos continuam funcionando normalmente."""
    representation = Representation([ApneaDomain(), HypoxiaDomain()])
    assert representation.domain_names() == ["apnea", "hypoxia"]


def test_representation_space_empty_matrix_raises_clear_error():
    """Um RepresentationSpace vazio não deve falhar com um erro genérico do numpy — deve dizer claramente o que está errado."""
    space = RepresentationSpace()
    with pytest.raises(ValueError, match="vazio"):
        space.matrix()


def test_representation_space_get_missing_id_raises_clear_error():
    space = RepresentationSpace()
    with pytest.raises(KeyError, match="não está neste RepresentationSpace"):
        space.get("id_que_nao_existe")


def test_trajectory_latest_on_empty_trajectory_raises_clear_error():
    """Antes: IndexError genérico ('list index out of range'). Agora: mensagem aponta a causa (trajetória vazia)."""
    traj = Trajectory(system_id="paciente_fantasma")
    with pytest.raises(IndexError, match="vazia"):
        traj.latest()


def test_normal_distribution_rejects_negative_std():
    """std negativo não tem significado estatístico — deve falhar na construção, não dentro de numpy.random mais tarde."""
    with pytest.raises(ValueError, match="negativo"):
        Normal(mean=82.0, std=-5.0)


def test_normal_distribution_accepts_zero_std():
    """std=0.0 é uma distribuição degenerada válida (equivalente a um valor pontual) — não deve ser rejeitado."""
    d = Normal(mean=82.0, std=0.0)
    assert d.std == 0.0


def test_composite_representation_requires_name():
    with pytest.raises(ValueError, match="name"):
        CompositeRepresentation(name="", children=[ApneaDomain()])


def test_composite_representation_requires_nonempty_children():
    with pytest.raises(ValueError, match="filho"):
        CompositeRepresentation(name="grupo_vazio", children=[])
