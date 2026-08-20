from __future__ import annotations

import json
import platform
from typing import Any, Dict

from .optional_deps import availability_report


def collect_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        info["torch"] = None
        info["cuda"] = None
        info["cuda_available"] = False
        info["gpu_name"] = None
        info["torch_error"] = repr(exc)
    info["optional_dependencies"] = availability_report()
    return info


def main() -> None:
    print(json.dumps(collect_env(), indent=2))


if __name__ == "__main__":
    main()
