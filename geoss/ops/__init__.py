"""Optional fused operators with numerically equivalent PyTorch fallbacks."""

from .flow_matching import construct_flow_training_pair, flow_matching_pair, fused_flow_matching_available

__all__ = ["construct_flow_training_pair", "flow_matching_pair", "fused_flow_matching_available"]
