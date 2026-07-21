# src/models/torchseg/postprocessor.py: reuse the seg mask extraction for torchseg

from src.models.seg.postprocessor import SegPostprocessor


class TorchSegPostprocessor(SegPostprocessor):
    """Reuses SegPostprocessor mask-to-corner extraction for torchseg models."""
