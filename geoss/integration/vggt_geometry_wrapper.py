from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGTGeometryWrapper(nn.Module):
    """Frozen VGGT geometry extractor with mock fallback for dry-runs."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        vggt_root: Optional[str] = None,
        checkpoint: Optional[str] = None,
        pretrained_name: Optional[str] = None,
        mock: bool = False,
        cache_features: bool = False,
        vggt_image_size: int = 518,
    ) -> None:
        super().__init__()
        self.mock = mock
        self.cache_features = cache_features
        self.vggt_image_size = vggt_image_size
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.model = model
        self.pose_decoder = None
        if self.model is None and not mock:
            self.model, self.pose_decoder = self._load_vggt(vggt_root, checkpoint, pretrained_name)
        elif self.model is not None:
            self.pose_decoder = self._load_pose_decoder(vggt_root)
        if self.model is not None:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor, *, use_cache: bool = False) -> Dict[str, torch.Tensor]:
        images = self._normalize_input_images(images)
        if self.mock or self.model is None:
            return self._mock_forward(images)
        cache_key = (int(images.data_ptr()), tuple(images.shape), str(images.device))
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        model_images = _resize_for_vggt(images, self.vggt_image_size)
        with torch.no_grad():
            predictions, tokens, patch_start_idx, raw_last_shape = self._forward_vggt_once(model_images)
            out = self._normalize_predictions(predictions, model_images, output_size=images.shape[-2:])
            if tokens is not None:
                dense = _tokens_to_spatial_features(tokens, model_images.shape[-2:])
                out["vggt_features"] = dense if dense is not None else tokens
                out["vggt_feature_tokens"] = tokens
                out["feature_shape_info"] = {
                    "format": "patch_grid_from_tokens" if dense is not None else "aggregator_patch_tokens",
                    "tokens": list(tokens.shape),
                    "features": list(dense.shape) if dense is not None else None,
                    "patch_start_idx": int(patch_start_idx),
                    "raw_last_tokens": raw_last_shape,
                    "model_image_size": list(model_images.shape[-2:]),
                    "original_image_size": list(images.shape[-2:]),
                }
            else:
                out["vggt_features"] = None
                out["feature_shape_info"] = {"format": "unavailable", "model_image_size": list(model_images.shape[-2:])}
        if self.cache_features or use_cache:
            self._cache[cache_key] = out
        return out

    def _forward_vggt_once(self, images: torch.Tensor):
        if not hasattr(self.model, "aggregator"):
            predictions = self.model(images)
            return predictions, None, 0, None
        aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
        predictions: Dict[str, torch.Tensor] = {}
        with torch.cuda.amp.autocast(enabled=False):
            if getattr(self.model, "camera_head", None) is not None:
                pose_enc_list = self.model.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
            if getattr(self.model, "depth_head", None) is not None:
                depth, depth_conf = self.model.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
            if getattr(self.model, "point_head", None) is not None:
                pts3d, pts3d_conf = self.model.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
        last = aggregated_tokens_list[-1]
        tokens = last[:, :, patch_start_idx:]
        return predictions, tokens, patch_start_idx, list(last.shape)

    def _normalize_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        images: torch.Tensor,
        *,
        output_size: Optional[tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, _, H, W = images.shape
        out_h, out_w = output_size or (H, W)
        out: Dict[str, Any] = {}
        depth = _first_present(predictions, ["depth", "depth_map", "pred_depth", "vggt_depth"])
        out["vggt_depth"] = _resize_b_n_c_h_w(_to_b_n_1_h_w(depth, B, N), (out_h, out_w)) if depth is not None else None
        pointmap = _first_present(predictions, ["world_points", "point_map", "pointmap", "points3d", "vggt_pointmap"])
        out["vggt_pointmap"] = _resize_b_n_c_h_w(_to_b_n_3_h_w(pointmap, B, N), (out_h, out_w)) if pointmap is not None else None
        conf = _first_present(predictions, ["depth_conf", "world_points_conf", "confidence", "conf"])
        out["vggt_confidence"] = _resize_confidence(conf, B, N, (out_h, out_w)) if conf is not None else None
        pose = _first_present(predictions, ["pose_enc", "camera", "camera_pose"])
        if pose is not None and self.pose_decoder is not None:
            extrinsic, intrinsic = self.pose_decoder(pose, images.shape[-2:])
            w2c = torch.eye(4, device=images.device, dtype=images.dtype).view(1, 1, 4, 4).repeat(B, N, 1, 1)
            w2c[:, :, :3, :4] = extrinsic.to(images.dtype)
            K = intrinsic.to(images.dtype)
            if (out_h, out_w) != (H, W):
                K = K.clone()
                K[..., 0, :] *= float(out_w) / float(W)
                K[..., 1, :] *= float(out_h) / float(H)
            out["vggt_camera"] = {"w2c": w2c, "c2w": torch.linalg.inv(w2c), "K": K}
        out.setdefault("feature_shape_info", {})
        return out

    def _mock_forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, _, H, W = images.shape
        pooled = F.interpolate(images.reshape(B * N, 3, H, W), size=(H // 8, W // 8), mode="bilinear", align_corners=False)
        features = pooled.reshape(B, N, 3, H // 8, W // 8)
        depth = torch.ones(B, N, 1, H, W, device=images.device, dtype=images.dtype)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=images.device, dtype=images.dtype),
            torch.linspace(-1, 1, W, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        pointmap = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).view(1, 1, 3, H, W).expand(B, N, -1, -1, -1)
        return {
            "vggt_features": features,
            "vggt_depth": depth,
            "vggt_pointmap": pointmap,
            "vggt_confidence": torch.ones(B, N, H, W, device=images.device, dtype=images.dtype),
            "feature_shape_info": {"format": "mock_spatial", "features": list(features.shape)},
        }

    @staticmethod
    def _normalize_input_images(images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5:
            raise ValueError(f"VGGTGeometryWrapper expects images [B,N,3,H,W] or [N,3,H,W], got {tuple(images.shape)}")
        if images.shape[2] != 3:
            raise ValueError(f"VGGTGeometryWrapper expects RGB channel at dim 2, got {tuple(images.shape)}")
        return images

    def _load_pose_decoder(self, vggt_root: Optional[str]):
        if vggt_root:
            sys.path.insert(0, str(Path(vggt_root)))
        try:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except Exception:
            return None
        return pose_encoding_to_extri_intri

    def _load_vggt(self, vggt_root: Optional[str], checkpoint: Optional[str], pretrained_name: Optional[str]):
        if vggt_root:
            sys.path.insert(0, str(Path(vggt_root)))
        try:
            from vggt.models.vggt import VGGT
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except Exception as exc:
            raise ImportError("Could not import VGGT. Use mock=True for dry-run.") from exc
        if pretrained_name is not None and hasattr(VGGT, "from_pretrained"):
            try:
                model = VGGT.from_pretrained(pretrained_name)
            except Exception:
                model = VGGT()
                _load_vggt_pretrained_fallback(model, pretrained_name)
        else:
            model = VGGT()
            if pretrained_name is not None:
                _load_vggt_pretrained_fallback(model, pretrained_name)
        if checkpoint:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
            state = torch.load(ckpt_path, map_location="cpu")
            state = _extract_state_dict(state)
            model.load_state_dict(state, strict=False)
        return model, pose_encoding_to_extri_intri


def _to_b_n_1_h_w(depth: torch.Tensor, B: int, N: int) -> torch.Tensor:
    if depth is None:
        return None
    if depth.shape[:3] == (B, N, 1):
        return depth
    if depth.shape[0] == B and depth.shape[1] == N and depth.shape[-1] == 1:
        return depth.permute(0, 1, 4, 2, 3).contiguous()
    raise ValueError(f"Unsupported VGGT depth shape {tuple(depth.shape)}")


def _to_b_n_3_h_w(points: torch.Tensor, B: int, N: int) -> torch.Tensor:
    if points.shape[:3] == (B, N, 3):
        return points
    if points.shape[0] == B and points.shape[1] == N and points.shape[-1] == 3:
        return points.permute(0, 1, 4, 2, 3).contiguous()
    raise ValueError(f"Unsupported VGGT pointmap shape {tuple(points.shape)}")


def _resize_for_vggt(images: torch.Tensor, target_size: int) -> torch.Tensor:
    if target_size <= 0:
        target_size = _round_up_to_multiple(max(images.shape[-2:]), 14)
    if images.shape[-1] == target_size and images.shape[-2] == target_size:
        return images
    B, N, C, H, W = images.shape
    resized = F.interpolate(
        images.reshape(B * N, C, H, W),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(B, N, C, target_size, target_size).contiguous()


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return int((value + multiple - 1) // multiple * multiple)


def _resize_b_n_c_h_w(tensor: Optional[torch.Tensor], output_size: tuple[int, int]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.shape[-2:] == output_size:
        return tensor
    B, N, C, H, W = tensor.shape
    resized = F.interpolate(
        tensor.reshape(B * N, C, H, W).float(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.to(dtype=tensor.dtype).reshape(B, N, C, output_size[0], output_size[1]).contiguous()


def _resize_confidence(conf: torch.Tensor, B: int, N: int, output_size: tuple[int, int]) -> torch.Tensor:
    if conf.shape[:2] == (B, N) and conf.ndim == 4:
        conf = conf.unsqueeze(2)
    elif conf.shape[:3] == (B, N, 1):
        conf = conf.contiguous()
    elif conf.shape[0] == B and conf.shape[1] == N and conf.shape[-1] == 1:
        conf = conf.permute(0, 1, 4, 2, 3).contiguous()
    else:
        return conf
    resized = _resize_b_n_c_h_w(conf, output_size)
    return resized[:, :, 0] if resized is not None else conf[:, :, 0]


def _tokens_to_spatial_features(tokens: torch.Tensor, model_size: tuple[int, int]) -> Optional[torch.Tensor]:
    if tokens.ndim != 4:
        return None
    B, N, T, C = tokens.shape
    side = int(T ** 0.5)
    if side * side != T:
        h = max(1, int(round(model_size[0] / 14)))
        w = max(1, T // h)
        if h * w != T:
            return None
    else:
        h = w = side
    return tokens.permute(0, 1, 3, 2).reshape(B, N, C, h, w).contiguous()


def _first_present(mapping: Dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model", "model_state_dict", "module"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return {k.removeprefix("module."): v for k, v in value.items()}
    return {k.removeprefix("module."): v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}


def _load_vggt_pretrained_fallback(model: nn.Module, pretrained_name: str) -> None:
    """Mirror VGGT demo loading for environments where from_pretrained is unavailable."""
    aliases = {
        "facebook/VGGT-1B": "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        "VGGT-1B": "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
    }
    url = aliases.get(pretrained_name, pretrained_name)
    if url.startswith("http://") or url.startswith("https://"):
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    else:
        path = Path(url)
        if not path.exists():
            raise FileNotFoundError(
                f"VGGT pretrained source not found: {pretrained_name}. "
                "Use --vggt_pretrained facebook/VGGT-1B or --vggt_checkpoint /path/to/model.pt."
            )
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(_extract_state_dict(state), strict=False)
