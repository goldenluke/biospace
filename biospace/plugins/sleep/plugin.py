"""biospace.plugins.sleep.plugin — registro deste módulo como Plugin do ecossistema."""

from __future__ import annotations

from biospace.core import Plugin

__all__ = ["SleepPlugin"]


class SleepPlugin(Plugin):
    name = "sleep"

    def describe(self) -> str:
        return (
            "Plugin de representação computacional para Síndrome da Apneia Obstrutiva "
            "do Sono (SAOS), migrado da pipeline real de pré-processamento em pandas: "
            "domínios de antropometria, apneia (via IDO), hipoxemia, arquitetura do sono, "
            "cardiovascular, comorbidades, sintomas e tratamento."
        )
