"""
scripts/check_dashboard_uci.py
=================================

Mesmo papel de scripts/check_dashboard.py, para o dashboard UCI
Diabetes 130-US Hospitals. Usa amostra reduzida (5.000 encontros) para
verificação rápida. Pula com aviso claro se o arquivo real não estiver
disponível.

Rode manualmente: python3 scripts/check_dashboard_uci.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_ROOT / "biospace_dashboard_uci"
CAMINHO_CSV = "/mnt/user-data/uploads/diabetic_data.csv"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DASHBOARD_DIR))


def main() -> int:
    if not os.path.exists(CAMINHO_CSV):
        print(f"Arquivo UCI ausente em {CAMINHO_CSV} — pulando verificação.")
        return 0

    os.chdir(DASHBOARD_DIR)

    import components._bootstrap  # noqa: F401
    from components.pipeline import run_pipeline
    from streamlit.testing.v1 import AppTest

    print("Carregando amostra UCI real (5.000 encontros)...")
    pipeline = run_pipeline(CAMINHO_CSV, max_rows=5000)
    print(f"  {pipeline.n_pacientes} pacientes carregados.\n")

    page_files = ["App.py"] + sorted(f"pages/{p}" for p in os.listdir("pages") if p.endswith(".py"))

    n_erros = 0
    for page in page_files:
        at = AppTest.from_file(page)
        at.session_state["biospace_uci_pipeline"] = pipeline
        at.session_state["biospace_uci_pipeline_source"] = "real::amostra5000"
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
