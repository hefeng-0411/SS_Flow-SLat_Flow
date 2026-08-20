from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict

from geoss.utils.optional_deps import require_dependency


def run_splatfacto_baseline(dataset_dir: str | Path, output_dir: str | Path, *, real_mode: bool = False) -> Dict[str, str]:
    require_dependency("nerfstudio", real_mode=real_mode, feature="Nerfstudio/Splatfacto baseline")
    cmd = ["ns-train", "splatfacto", "--data", str(dataset_dir), "--output-dir", str(output_dir)]
    subprocess.run(cmd, check=True)
    return {"command": " ".join(cmd), "output_dir": str(output_dir)}
