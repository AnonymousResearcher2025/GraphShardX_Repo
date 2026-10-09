"""GraphShardX: immutable vectors, decentralized certification, similarity-aware shards."""

from .config import Settings
from .model import GraphShardXError, IntegrityError, Unavailable

__all__ = ["Settings", "GraphShardXError", "IntegrityError", "Unavailable"]
