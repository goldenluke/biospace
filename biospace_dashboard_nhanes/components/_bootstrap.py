"""
components._bootstrap
========================

Garante que o pacote `biospace` (sibling do diretório do dashboard) esteja
importável, independentemente de onde o Streamlit for iniciado.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _DASHBOARD_DIR.parent

for candidate in (_PARENT_DIR, _DASHBOARD_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
