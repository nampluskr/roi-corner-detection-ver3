# src/models/hybrid/preprocessor.py: rasterize standard corners into a filled quad mask target

from src.models.seg.preprocessor import SegPreprocessor


class HybridPreprocessor(SegPreprocessor):
    """Reuses the seg mask rasterization: (N, 4, 2) corners into a (N, 1, mask_size, mask_size) binary mask."""
