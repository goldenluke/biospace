from .dynamic_system import DynamicSystem
from .evolution import EvolutionOperator, FeatureDynamics, MeanRevertingEvolutionOperator
from .stability import RobustnessReport, StabilityOperator, StabilityReport, check_feature_stability_robustness

__all__ = [
    "DynamicSystem",
    "EvolutionOperator",
    "MeanRevertingEvolutionOperator",
    "FeatureDynamics",
    "StabilityOperator",
    "StabilityReport",
    "RobustnessReport",
    "check_feature_stability_robustness",
]
# Nota: alerta precoce / critical slowing down NÃO fica aqui -- ver
# `biospace.early_warning.CriticalSlowingDownDetector`, o detector
# completo (3 indicadores, tendência de Kendall, significância por
# substitutos AR(1)). Uma versão mais simples (`compute_early_warning_indicators`)
# foi construída aqui por engano, sem saber que a versão completa já
# existia — removida numa auditoria posterior, não deixada como
# alternativa "mais simples".
