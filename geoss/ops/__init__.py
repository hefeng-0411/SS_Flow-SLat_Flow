"""Optional fused operators with numerically equivalent PyTorch fallbacks."""

from .flow_matching import flow_matching_pair, fused_flow_matching_available

__all__ = ["flow_matching_pair", "fused_flow_matching_available"]
