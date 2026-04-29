from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

from api.local_write_guard import install_production_write_guard

install_production_write_guard(PROJECT_ROOT)
