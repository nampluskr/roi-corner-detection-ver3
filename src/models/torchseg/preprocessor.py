# src/models/torchseg/preprocessor.py: reuse the seg mask target for torchseg

from src.models.seg.preprocessor import SegPreprocessor


class TorchSegPreprocessor(SegPreprocessor):
    """Reuses SegPreprocessor mask rasterization for torchseg models."""
