"""
scripts/check_dashboard_nhanes.py
====================================

Mesmo papel de scripts/check_dashboard.py, para o dashboard NHANES.
Requer os 6 arquivos .XPT reais em /mnt/user-data/uploads — pula com
aviso claro se ausentes (dado real, não gerado sinteticamente).

Rode manualmente: python3 scripts/check_dashboard_nhanes.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_ROOT / "biospace_dashboard_nhanes"
DADOS_DIR = "/mnt/user-data/uploads"
ARQUIVOS_NECESSARIOS = ["P_DEMO.xpt", "P_GHB.xpt", "P_GLU.xpt", "P_BMX.xpt", "P_BPXO.xpt", "P_DIQ.xpt"]

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DASHBOARD_DIR))


def main() -> int:
    faltando = [f for f in ARQUIVOS_NECESSARIOS if not os.path.exists(os.path.join(DADOS_DIR, f))]
    if faltando:
        print(f"Arquivos NHANES ausentes em {DADOS_DIR}: {faltando} — pulando verificação.")
        return 0

    os.chdir(DASHBOARD_DIR)

    import components._bootstrap  # noqa: F401
    from components.pipeline import run_pipeline
    from streamlit.testing.v1 import AppTest

    print("Carregando coorte NHANES real...")
    pipeline = run_pipeline(DADOS_DIR, idade_minima=20)
    print(f"  {pipeline.n_adultos} adultos carregados.\n")

    page_files = ["App.py"] + sorted(f"pages/{p}" for p in os.listdir("pages") if p.endswith(".py"))

    n_erros = 0
    for page in page_files:
        at = AppTest.from_file(page)
        at.session_state["biospace_nhanes_pipeline"] = pipeline
        at.session_state["biospace_nhanes_pipeline_source"] = "real::adultos"
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
