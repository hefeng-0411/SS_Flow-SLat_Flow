from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from geoss.utils.optional_deps import require_dependency


class HLocPipeline:
    def __init__(self, *, real_mode: bool = False) -> None:
        self.real_mode = real_mode
        require_dependency("hloc", real_mode=real_mode, feature="hloc feature extraction/matching")
        require_dependency("pycolmap", real_mode=real_mode, feature="hloc COLMAP export")

    def extract_features(self, image_dir: str | Path, output_path: str | Path) -> Dict[str, str]:
        from hloc import extract_features

        conf = extract_features.confs["superpoint_aachen"]
        extract_features.main(conf, Path(image_dir), feature_path=Path(output_path))
        return {"features": str(output_path)}

    def match_pairs(self, pairs: str | Path, features: str | Path, output_path: str | Path) -> Dict[str, str]:
        from hloc import match_features

        conf = match_features.confs["superglue"]
        match_features.main(conf, Path(pairs), features=Path(features), matches=Path(output_path))
        return {"matches": str(output_path)}

    def run_geometric_verification(self, image_dir: str | Path, sfm_dir: str | Path, pairs: str | Path, features: str | Path, matches: str | Path):
        from hloc import reconstruction

        return reconstruction.main(Path(sfm_dir), Path(image_dir), Path(pairs), Path(features), Path(matches))

    def estimate_relative_pose(self, kpts0, kpts1, K0, K1):
        require_dependency("poselib", real_mode=self.real_mode, feature="relative pose RANSAC")
        import poselib

        return poselib.estimate_relative_pose(kpts0, kpts1, K0, K1)

    def export_to_colmap_format(self, *paths: Iterable[str]) -> Dict[str, list[str]]:
        return {"colmap_inputs": [str(p) for p in paths]}
