# src/methods/torchseg/postprocessor.py: reuse the seg mask extraction for torchseg

from src.methods.seg.postprocessor import SegPostprocessor


class TorchSegPostprocessor(SegPostprocessor):
    """Reuses SegPostprocessor mask-to-corner extraction for torchseg models."""
