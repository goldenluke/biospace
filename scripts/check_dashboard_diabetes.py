"""
scripts/check_dashboard_diabetes.py
======================================

Mesmo papel de `scripts/check_dashboard.py`, para o dashboard de
diabetes: roda todas as páginas via AppTest com uma coorte sintética
pequena, pega erros de importação/dessincronia antes de virarem um
deploy quebrado.

Rode manualmente: python3 scripts/check_dashboard_diabetes.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_ROOT / "biospace_dashboard_diabetes"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DASHBOARD_DIR))


def main() -> int:
    os.chdir(DASHBOARD_DIR)

    import components._bootstrap  # noqa: F401
    from components.pipeline import run_pipeline
    from streamlit.testing.v1 import AppTest

    from biospace.plugins.diabetes import generate_synthetic_dataframe

    print("Gerando coorte sintética de verificação...")
    df = generate_synthetic_dataframe(n_per_group=15, seed=0)
    pipeline = run_pipeline(df)
    print(f"  {len(pipeline.cohort)} pacientes sintéticos gerados.\n")

    page_files = ["App.py"] + sorted(f"pages/{p}" for p in os.listdir("pages") if p.endswith(".py"))

    n_erros = 0
    for page in page_files:
        at = AppTest.from_file(page)
        at.session_state["biospace_diabetes_pipeline"] = pipeline
        at.session_state["biospace_diabetes_pipeline_source"] = "synthetic::15"
        try:
            at.run(timeout=180)
        except Exception as e:
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
