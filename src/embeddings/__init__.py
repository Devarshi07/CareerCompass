# src/embeddings/__init__.py
"""Embedding generation module - OpenAI only"""
from .embedding_generator import EmbeddingGenerator, get_embeddings

__all__ = ['EmbeddingGenerator', 'get_embeddings']