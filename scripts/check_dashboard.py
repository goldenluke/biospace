"""
scripts/check_dashboard.py
=============================

Roda TODAS as páginas do dashboard (App.py + pages/*.py) via
`streamlit.testing.v1.AppTest`, com uma coorte sintética pequena e
rápida — pega exatamente a classe de erro que já quebrou um deploy real
no Streamlit Cloud (ImportError por um arquivo desatualizado em relação
a outro, dependência faltando, etc.) ANTES de chegar em produção.

Rode manualmente: python3 scripts/check_dashboard.py
Ou via CI: já integrado em .github/workflows/ci.yml

Sai com código 0 se tudo passar, 1 se qualquer página falhar (para o CI
conseguir bloquear o merge/deploy).
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_ROOT / "biospace_dashboard"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DASHBOARD_DIR))


def main() -> int:
    os.chdir(DASHBOARD_DIR)

    import components._bootstrap  # noqa: F401
    from components.pipeline import run_pipeline
    from components.synthetic import generate_synthetic_dataframe
    from streamlit.testing.v1 import AppTest

    print("Gerando coorte sintética de verificação...")
    df = generate_synthetic_dataframe(n_per_group=15, seed=0)
    pipeline = run_pipeline(df)
    print(f"  {len(pipeline.cohort)} pacientes sintéticos gerados.\n")

    page_files = ["App.py"] + sorted(f"pages/{p}" for p in os.listdir("pages") if p.endswith(".py"))

    n_erros = 0
    for page in page_files:
        at = AppTest.from_file(page)
        at.session_state["biospace_pipeline"] = pipeline
        at.session_state["biospace_pipeline_source"] = "synthetic::15"
        try:
            at.run(timeout=180)
        except Exception as e:  # falha ao nível do próprio AppTest, não só exceção capturada na página
            print(f"ERRO  {page}: {e}")
            n_erros += 1
            continue

        if at.exception:
            n_erros += 1
            print(f"ERRO  {page}")
            for exc in at.exception:
                print(f"        {exc}")
        else:
            print(f"OK    {page}")

    print(f"\n{'='*50}\n{len(page_files) - n_erros}/{len(page_files)} páginas OK.")
    return 1 if n_erros else 0


if __name__ == "__main__":
    sys.exit(main())
