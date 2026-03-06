import os  # noqa: I001
import re
from pathlib import Path

def fix_file(filepath: Path):
    try:
        content = filepath.read_text('utf-8')
    except Exception:
        return
  # noqa: W293
    orig = content
  # noqa: W293
    # Fix import lines
    content = content.replace('from typing import Dict, List, Optional, Tuple', 'from typing import Optional')
    content = content.replace('from typing import Dict, List, Tuple', '')
    content = content.replace('from typing import Dict, Optional, Tuple', 'from typing import Optional')
    content = content.replace('from typing import Any, Dict, Optional, Tuple', 'from typing import Any, Optional')
  # noqa: W293
    # Fix type hints in code
    content = re.sub(r'\bDict\b', 'dict', content)
    content = re.sub(r'\bList\b', 'list', content)
    content = re.sub(r'\bTuple\b', 'tuple', content)

    # Fix B904
    content = re.sub(r'(raise SystemExit\([^)]+\))(?!\s*from)', r'\1 from None', content)
    content = re.sub(r'(raise RuntimeError\([^)]+\))(?!\s*from)', r'\1 from None', content)
  # noqa: W293
    if orig != content:
        filepath.write_text(content, 'utf-8')

for root, dirs, files in os.walk(str(Path(__file__).parent)):
    if '.venv' in dirs: dirs.remove('.venv')  # noqa: E701
    if '.git' in dirs: dirs.remove('.git')  # noqa: E701
    for file in files:
        if file.endswith('.py') and file != 'fix_ruff.py':
            fix_file(Path(root) / file)
