"""
biospace.geometry.curvature
==============================

Fase 8 — Geometria:

    Paciente -> Representação -> Variedade -> Trajetória -> Curvatura -> Estabilidade

Duas formas INDEPENDENTES de estimar curvatura, propositalmente
cruzadas uma contra a outra (ver `tests/test_curvature.py`):

  1. `FeatureDynamics.curvature` (em `biospace.dynamics`) — vem do φ já
     ajustado longitudinalmente (dinâmica temporal real de cada
     paciente ao longo do tempo).
  2. `estimate_density_curvature` (aqui) — vem da FORMA da distribuição
     POPULACIONAL transversal (um instante só, muitos pacientes) via
     reconstrução de um "potencial efetivo" U(x) = -log(densidade(x)) —
     técnica padrão em reconstrução de paisagem (Waddington-landscape,
     usada em biologia computacional para diferenciação celular;
     "quasi-potencial" em ecologia/clima para mudanças de regime).

Se as duas concordarem (mesmo calculadas de formas totalmente
diferentes, uma temporal, outra transversal), isso é evidência de que
"curvatura" está capturando algo real sobre a estrutura do sistema, não
um artefato de um método específico.

`detect_metastability`: conta quantos POÇOS distintos existem na
paisagem — mais de um poço = MÚLTIPLOS estados estáveis (metaestabilidade
de verdade — não apenas "há uma clusterização", mas "existe uma barreira
de energia real entre os grupos", quantificada pela altura da barreira).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["estimate_density_curvature", "detect_metastability", "MetastabilityReport", "Well"]


def _feature_values(space: "RepresentationSpace", feature_name: str, order: Optional[Sequence[str]] = None) -> np.ndarray:
    """Extrai os valores de UMA Feature (nome qualificado 'dominio.feature') para toda a população de `space`."""
    used_order = order or space.order()
    domain_name, _, atom_name = feature_name.partition(".")
    values = []
    for sid in space.ids():
        vec = space.get(sid)
        f = next((f for f in vec.components.get(domain_name, []) if f.name == atom_name), None)
        if f is not None:
            values.append(f.value)
    return np.array(values)


def _effective_potential(values: np.ndarray, n_grid: int = 500, bandwidth: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """U(x) = -log(densidade(x)) sobre uma grade fina cobrindo o alcance observado de `values`."""
    kde = gaussian_kde(values, bw_method=bandwidth)
    span = values.max() - values.min()
    grid = np.linspace(values.min() - 0.1 * span, values.max() + 0.1 * span, n_grid)
    density = kde(grid)
    density = np.clip(density, 1e-300, None)
    U = -np.log(density)
    return grid, U


def estimate_density_curvature(
    space: "RepresentationSpace", feature_name: str, order: Optional[Sequence[str]] = None, bandwidth: Optional[float] = None
) -> float:
    """
    Curvatura no MODO da distribuição populacional (o poço mais
    profundo): U''(x) no ponto de densidade máxima, estimado por
    diferença finita central sobre uma reconstrução por KDE de
    U(x) = -log(densidade(x)).

    AVISO — LIMITAÇÃO REAL ENCONTRADA NA VALIDAÇÃO (ver README/testes):
    diferenciar numericamente uma curva de KDE DUAS VEZES (2ª derivada)
    amplifica MUITO qualquer imprecisão da estimativa de densidade — um
    problema estatístico bem documentado, não um bug específico desta
    implementação. Testado em cenário sintético com curvatura verdadeira
    IDÊNTICA entre 4 Features (variância estacionária construída para
    ser igual): o valor retornado variou quase 5x entre elas (0,14 a
    0,71), mesmo com bandwidth fixo (afastando viés de seleção
    automática) e n=300. **Use este valor como indicador QUALITATIVO
    (ordem de grandeza, comparação grosseira), não para comparação
    numérica precisa contra `FeatureDynamics.curvature`** (que, em
    contraste, recuperou o k verdadeiro quase exatamente no mesmo
    cenário). Para contagem de poços e detecção de metaestabilidade
    (`detect_metastability`), a limitação é bem menor — encontrar PICOS
    de densidade é muito mais robusto que estimar a curvatura exata
    no pico.
    """
    values = _feature_values(space, feature_name, order)
    if len(values) < 5:
        raise ValueError(f"Poucos pontos ({len(values)}) para estimar densidade de forma confiável (mínimo razoável: 5).")

    grid, U = _effective_potential(values, bandwidth=bandwidth)
    idx_min = int(np.argmin(U))
    idx_min = min(max(idx_min, 1), len(grid) - 2)
    h = grid[1] - grid[0]
    curvature = (U[idx_min + 1] - 2 * U[idx_min] + U[idx_min - 1]) / (h**2)
    return float(curvature)


@dataclass
class Well:
    """Um poço (estado metaestável) na paisagem de potencial de uma Feature."""

    position: float
    depth: float
    barrier_left: Optional[float] = None
    barrier_right: Optional[float] = None

    @property
    def escape_barrier(self) -> Optional[float]:
        """A MENOR das duas barreiras vizinhas — a rota de escape mais fácil para fora deste poço."""
        candidatos = [b for b in (self.barrier_left, self.barrier_right) if b is not None]
        return min(candidatos) if candidatos else None


@dataclass
class MetastabilityReport:
    feature_name: str
    wells: list[Well] = field(default_factory=list)

    @property
    def n_wells(self) -> int:
        return len(self.wells)

    @property
    def is_metastable(self) -> bool:
        """Mais de um poço = existe mais de um estado estável — metaestabilidade genuína."""
        return self.n_wells >= 2

    def summary(self) -> str:
        lines = [f"Paisagem de '{self.feature_name}': {self.n_wells} poço(s) — {'METAESTÁVEL' if self.is_metastable else 'unimodal'}"]
        for i, well in enumerate(self.wells):
            barreira = f"escape_barrier={well.escape_barrier:.3f}" if well.escape_barrier is not None else "poço único"
            lines.append(f"  Poço {i + 1}: posição={well.position:.3f}, profundidade(U)={well.depth:.3f}, {barreira}")
        return "\n".join(lines)


def detect_metastability(
    space: "RepresentationSpace",
    feature_name: str,
    order: Optional[Sequence[str]] = None,
    bandwidth: Optional[float] = None,
    min_prominence: float = 0.3,
) -> MetastabilityReport:
    """
    Conta quantos poços distintos existem na paisagem de potencial
    U(x) = -log(densidade(x)) de UMA Feature, e a altura da barreira
    entre poços vizinhos. `min_prominence`: proeminência mínima (em
    unidades de U — nats de log-densidade) para um poço ser considerado
    genuíno, não ruído de amostragem do KDE (`scipy.signal.find_peaks`,
    aplicado a -U, ou seja, aos PICOS de densidade).
    """
    values = _feature_values(space, feature_name, order)
    if len(values) < 10:
        raise ValueError(f"Poucos pontos ({len(values)}) para detectar metaestabilidade de forma confiável (mínimo razoável: 10).")

    grid, U = _effective_potential(values, bandwidth=bandwidth)
    neg_U = -U

    peaks, _ = find_peaks(neg_U, prominence=min_prominence)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmin(U))])

    wells = []
    for i, peak_idx in enumerate(peaks):
        barrier_left = None
        barrier_right = None
        if i > 0:
            barrier_left = float(np.max(U[peaks[i - 1] : peak_idx + 1]) - U[peak_idx])
        if i < len(peaks) - 1:
            barrier_right = float(np.max(U[peak_idx : peaks[i + 1] + 1]) - U[peak_idx])
        wells.append(Well(position=float(grid[peak_idx]), depth=float(U[peak_idx]), barrier_left=barrier_left, barrier_right=barrier_right))

    return MetastabilityReport(feature_name=feature_name, wells=wells)
