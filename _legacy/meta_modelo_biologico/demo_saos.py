"""
demo_saos.py
============

Aplica o meta-modelo (metamodel.py) ao caso de validação usado no artigo:
Síndrome da Apneia Obstrutiva do Sono (SAOS).

Reproduz, com dados sintéticos, os conceitos centrais do texto:

  - R1(P) = (IAH)                              -> representação pobre
  - R2(P) = (IAH, IMC, Idade)                   -> representação intermediária
  - R3(P) = (IAH, ODI, SpO2min, ...)             -> representação rica

  - domínios semânticos independentes (respiração, hipóxia, antropometria)
  - composição R(B) = (φ_resp(D_resp), φ_hip(D_hip), φ_antropo(D_antropo))
  - espaço de representação X e geometria (distância entre pacientes)
  - fenotipagem como região do espaço, estimada via KMeans
  - coorte longitudinal: cada novo exame apenas atualiza a trajetória
  - matriz de transição fenotípica
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta

from metamodel import (
    BiologicalSystem,
    Observation,
    SemanticDomain,
    RepresentationOperator,
    Representation,
    RepresentationSpace,
    Geometry,
    Cohort,
    KMeansPhenotypeEstimator,
    Contracts,
)


# -----------------------------------------------------------------------------
# 1. Geração sintética de pacientes com SAOS
# -----------------------------------------------------------------------------
rng = np.random.default_rng(42)


def make_patient(patient_id: str, severity: str) -> BiologicalSystem:
    """
    Cria um paciente sintético com um perfil de severidade
    ('leve', 'moderada', 'grave'), simulando dados de polissonografia,
    antropometria e oximetria.
    """
    profiles = {
        "leve": dict(iah=(6, 2), rdi=(8, 2), imc=(24, 2), idade=(35, 8), spo2min=(90, 2), odi=(7, 2)),
        "moderada": dict(iah=(20, 4), rdi=(24, 4), imc=(29, 3), idade=(45, 8), spo2min=(85, 3), odi=(18, 4)),
        "grave": dict(iah=(45, 8), rdi=(50, 8), imc=(34, 4), idade=(52, 9), spo2min=(75, 5), odi=(40, 8)),
    }
    p = profiles[severity]
    raw = {
        "IAH": max(0, rng.normal(*p["iah"])),
        "RDI": max(0, rng.normal(*p["rdi"])),
        "IMC": max(15, rng.normal(*p["imc"])),
        "Idade": max(18, rng.normal(*p["idade"])),
        "SpO2min": min(100, max(50, rng.normal(*p["spo2min"]))),
        "ODI": max(0, rng.normal(*p["odi"])),
    }
    B = BiologicalSystem(patient_id=patient_id, metadata={"severity_true": severity})
    B.ingest(datetime.now(), raw)
    return B


# -----------------------------------------------------------------------------
# 2. Observações (operadores O_i : B -> M_i), com proveniência explícita
# -----------------------------------------------------------------------------
def latest(name):
    return lambda B: B.latest_raw_state()[name]


obs_iah = Observation("IAH", latest("IAH"), source="polissonografia", unit="eventos/hora")
obs_rdi = Observation("RDI", latest("RDI"), source="polissonografia", unit="eventos/hora")
obs_imc = Observation("IMC", latest("IMC"), source="antropometria", unit="kg/m2")
obs_idade = Observation("Idade", latest("Idade"), source="prontuario", unit="anos")
obs_spo2min = Observation("SpO2min", latest("SpO2min"), source="oximetria", unit="%")
obs_odi = Observation("ODI", latest("ODI"), source="oximetria", unit="eventos/hora")


# -----------------------------------------------------------------------------
# 3. Domínios semânticos (D_i = (Σ_i, O_i)) — Seção 4.3
# -----------------------------------------------------------------------------
dom_respiracao = SemanticDomain(
    name="respiracao",
    semantics="Eventos respiratórios obstrutivos durante o sono",
    observations=[obs_iah, obs_rdi],
)

dom_hipoxia = SemanticDomain(
    name="hipoxia",
    semantics="Grau de dessaturação de oxigênio associado aos eventos respiratórios",
    observations=[obs_spo2min, obs_odi],
)

dom_antropometria = SemanticDomain(
    name="antropometria",
    semantics="Características antropométricas e demográficas do paciente",
    observations=[obs_imc, obs_idade],
)


# -----------------------------------------------------------------------------
# 4. Operadores de representação (φ_i : D_i -> X_i) — Seção 4.4
#    Cada domínio escolhe sua própria função de representação (aqui, vetores
#    normalizados por z-score usando parâmetros populacionais de referência).
# -----------------------------------------------------------------------------
REF = {  # média, desvio (valores de referência populacionais, ilustrativos)
    "IAH": (15.0, 15.0),
    "RDI": (18.0, 16.0),
    "SpO2min": (85.0, 8.0),
    "ODI": (15.0, 14.0),
    "IMC": (27.0, 5.0),
    "Idade": (45.0, 12.0),
}


def zscore(name, value):
    mean, std = REF[name]
    return (value - mean) / std


phi_respiracao = RepresentationOperator(
    dom_respiracao,
    mapping=lambda obs: np.array([zscore("IAH", obs["IAH"].value), zscore("RDI", obs["RDI"].value)]),
)

phi_hipoxia = RepresentationOperator(
    dom_hipoxia,
    mapping=lambda obs: np.array([zscore("SpO2min", obs["SpO2min"].value), zscore("ODI", obs["ODI"].value)]),
)

phi_antropometria = RepresentationOperator(
    dom_antropometria,
    mapping=lambda obs: np.array([zscore("IMC", obs["IMC"].value), zscore("Idade", obs["Idade"].value)]),
)


# -----------------------------------------------------------------------------
# 5. Representação global R(B) = (φ_resp(D_resp), φ_hip(D_hip), φ_antropo(D_antropo))
# -----------------------------------------------------------------------------
R = Representation([phi_respiracao, phi_hipoxia, phi_antropometria])
DOMAIN_ORDER = ["respiracao", "hipoxia", "antropometria"]


# -----------------------------------------------------------------------------
# 6. Construção da coorte longitudinal (Seção 9)
# -----------------------------------------------------------------------------
cohort = Cohort(name="coorte_saos", representation=R)

severities = ["leve"] * 15 + ["moderada"] * 15 + ["grave"] * 15
patients = [make_patient(f"paciente_{i:03d}", sev) for i, sev in enumerate(severities)]

t0 = datetime.now()
for B in patients:
    cohort.add_observation(B, B.latest_raw_state(), timestamp=t0)

print(f"Coorte construída: {cohort}\n")

# Simula uma segunda visita (6 meses depois) para alguns pacientes, com leve
# melhora após tratamento — a trajetória é *atualizada*, não recriada.
t1 = t0 + timedelta(days=180)
for B in patients[:10]:
    raw = B.latest_raw_state().copy()
    raw["IAH"] = max(0, raw["IAH"] * 0.7)
    raw["ODI"] = max(0, raw["ODI"] * 0.7)
    raw["SpO2min"] = min(100, raw["SpO2min"] + 3)
    cohort.add_observation(B, raw, timestamp=t1)

print("Exemplo de trajetória (paciente_000):")
traj = cohort.trajectories["paciente_000"]
for i in range(len(traj)):
    rep = traj.at(i)
    print(f"  t={rep.timestamp.date()}  vetor={np.round(rep.as_vector(DOMAIN_ORDER), 2)}")
print()


# -----------------------------------------------------------------------------
# 7. Espaço de representação (X) e Geometria (G)
# -----------------------------------------------------------------------------
space = cohort.snapshot_space(order=DOMAIN_ORDER)
geometry = Geometry.euclidean()

M, ids = space.matrix()
print(f"Espaço de representação: {len(ids)} pacientes, dimensão {M.shape[1]}\n")

# Distância clínica entre dois pacientes quaisquer (Seção 7.1)
p_a, p_b = ids[0], ids[20]
d_ab = geometry.distance(space.get(p_a).as_vector(DOMAIN_ORDER), space.get(p_b).as_vector(DOMAIN_ORDER))
print(f"Distância (Euclidiana) entre {p_a} e {p_b}: {d_ab:.3f}\n")


# -----------------------------------------------------------------------------
# 8. Fenotipagem como região estimada do espaço (Seção 7.9 / 8)
# -----------------------------------------------------------------------------
estimator = KMeansPhenotypeEstimator(n_clusters=3)
labels = estimator.fit_predict(space, order=DOMAIN_ORDER)
phenotypes = estimator.to_phenotypes(space, labels, order=DOMAIN_ORDER)

print("Fenótipos estimados (regiões do espaço X):")
for ph in phenotypes:
    members = [pid for pid, lbl in labels.items() if f"{estimator.name}_cluster_{lbl}" == ph.name]
    true_sev = [cohort.patients[m].metadata["severity_true"] for m in members]
    from collections import Counter
    print(f"  {ph.name}: n={len(members)}, severidade real predominante={Counter(true_sev).most_common(1)}")
print()


# -----------------------------------------------------------------------------
# 9. Matriz de transição fenotípica (Seção 8.8 / 9.6)
#    (aqui com poucas trajetórias de 2 pontos, apenas ilustrativo)
# -----------------------------------------------------------------------------
P, names = cohort.transition_matrix(phenotypes, order=DOMAIN_ORDER)
print("Matriz de transição fenotípica P(F_j | F_i):")
print("           " + "  ".join(f"{n:>18}" for n in names))
for i, row in enumerate(P):
    print(f"{names[i]:>10}  " + "  ".join(f"{v:18.2f}" for v in row))
print()


# -----------------------------------------------------------------------------
# 10. Verificação de contratos formais (Seção 5) sobre esta implementação
# -----------------------------------------------------------------------------
B_test = patients[0]
print("Verificação de contratos formais:")
print(f"  Reprodutibilidade (5.8):        {Contracts.check_reproducibility(phi_respiracao, B_test)}")

B_other = patients[1]
print(
    "  Preservação semântica (5.2):    "
    f"{Contracts.check_semantic_preservation(phi_respiracao, B_test, B_other)}"
)

extra_obs = Observation("Idade2x", lambda B: B.latest_raw_state()['Idade'] * 2, source='sintetico')
extra_domain = SemanticDomain('idade_dobrada', 'domínio de teste', [extra_obs])
extra_operator = RepresentationOperator(extra_domain, mapping=lambda obs: np.array([obs['Idade2x'].value]))
print(f"  Extensibilidade (5.5):          {Contracts.check_extensibility(R, extra_operator, B_test)}")


def raw_distance(B1, B2):
    s1, s2 = B1.latest_raw_state(), B2.latest_raw_state()
    keys = ['IAH', 'RDI']
    return float(np.linalg.norm([s1[k] - s2[k] for k in keys]))


pairs = [(patients[i], patients[i + 1]) for i in range(0, 10)]
L = Contracts.check_lipschitz_continuity(phi_respiracao, pairs, raw_distance)
print(f"  Constante de Lipschitz estimada (5.4, continuidade): {L:.4f}")
