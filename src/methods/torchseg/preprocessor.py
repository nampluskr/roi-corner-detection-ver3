# src/methods/torchseg/preprocessor.py: reuse the seg mask target for torchseg

from src.methods.seg.preprocessor import SegPreprocessor


class TorchSegPreprocessor(SegPreprocessor):
    """Reuses SegPreprocessor mask rasterization for torchseg models."""
