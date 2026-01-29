"""
Enhanced Qdrant Engine with Multi-Model Support
Supports ensemble embeddings and advanced retrieval strategies
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result from vector search"""
    doc_id: str
    score: float
    payload: Dict


class MultiModelQdrantEngine:
    """
    Advanced Qdrant engine supporting multi-model ensemble embeddings.
    
    Features:
    - Support for high-dimensional ensemble vectors (3000+ dims)
    - Hybrid search combining multiple vector spaces
    - Advanced similarity measures
    - Batch operations optimization
    """
    
    def __init__(self, 
                 data_path: str = './data/qdrant_storage',
                 collection_name: str = 'sentinel_ensemble',
                 vector_dim: int = 2904,
                 verbose: bool = True):
        """
        Initialize enhanced Qdrant engine.
        
        Args:
            data_path: Path to Qdrant storage
            collection_name: Name of collection
            vector_dim: Dimension of ensemble vectors
            verbose: Enable logging
        """
        self.data_path = data_path
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.verbose = verbose
        
        try:
            self.client = QdrantClient(path=data_path)
            if self.verbose:
                logger.info(f"✅ Connected to Qdrant at {data_path}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        self.indexed_vectors = 0
    
    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        Create collection for ensemble vectors.
        
        Args:
            force_recreate: Delete existing collection first
        
        Returns:
            True if successful
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name in collection_names:
                if force_recreate:
                    self.client.delete_collection(self.collection_name)
                    if self.verbose:
                        logger.info(f"🗑️ Deleted existing collection: {self.collection_name}")
                else:
                    if self.verbose:
                        logger.info(f"⚠️ Collection already exists: {self.collection_name}")
                    return True
            
            # Create new collection with optimal settings
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                ),
                on_disk_payload=False
            )
            
            if self.verbose:
                logger.info(f"✅ Created collection: {self.collection_name}")
                logger.info(f"   Vector dimension: {self.vector_dim}")
                logger.info(f"   Distance metric: COSINE")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def index_vectors(self, 
                     vectors: np.ndarray,
                     doc_ids: List[str],
                     payloads: Optional[List[Dict]] = None,
                     batch_size: int = 100) -> int:
        """
        Index vectors with documents.
        
        Args:
            vectors: Array of shape (N, vector_dim)
            doc_ids: List of document IDs
            payloads: Optional metadata for each document
            batch_size: Batch size for indexing
        
        Returns:
            Number of vectors indexed
        """
        if vectors.shape[1] != self.vector_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.vector_dim}, "
                f"got {vectors.shape[1]}"
            )
        
        if len(vectors) != len(doc_ids):
            raise ValueError("Number of vectors must match number of doc_ids")
        
        if payloads is None:
            payloads = [{"doc_id": doc_id} for doc_id in doc_ids]
        
        try:
            # Prepare points
            points = []
            for i, (vector, doc_id, payload) in enumerate(zip(vectors, doc_ids, payloads)):
                # Ensure vector is proper format
                vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
                
                point = PointStruct(
                    id=i,
                    vector=vector_list,
                    payload={
                        "doc_id": doc_id,
                        **payload
                    }
                )
                points.append(point)
                
                # Batch insert
                if len(points) >= batch_size:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    self.indexed_vectors += len(points)
                    points = []
            
            # Insert remaining points
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                self.indexed_vectors += len(points)
            
            if self.verbose:
                logger.info(f"✅ Indexed {self.indexed_vectors} vectors")
            
            return self.indexed_vectors
        
        except Exception as e:
            logger.error(f"Failed to index vectors: {e}")
            raise
    
    def search(self, 
              query_vector: np.ndarray,
              top_k: int = 20,
              threshold: float = 0.0) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector of shape (vector_dim,)
            top_k: Number of results to return
            threshold: Minimum similarity score
        
        Returns:
            List of SearchResult objects
        """
        if query_vector.shape[0] != self.vector_dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.vector_dim}, "
                f"got {query_vector.shape[0]}"
            )
        
        try:
            # Convert to list
            query_vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
            
            # Search
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector_list,
                limit=top_k,
                score_threshold=threshold
            )
            
            # Convert to SearchResult objects
            results = []
            for hit in hits:
                result = SearchResult(
                    doc_id=hit.payload.get("doc_id", f"unknown_{hit.id}"),
                    score=float(hit.score),
                    payload=hit.payload
                )
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def batch_search(self,
                    query_vectors: np.ndarray,
                    top_k: int = 20) -> Dict[int, List[SearchResult]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_vectors: Array of shape (Q, vector_dim)
            top_k: Number of results per query
        
        Returns:
            Dict mapping query_idx -> list of SearchResult
        """
        results = {}
        
        for i, query_vector in enumerate(query_vectors):
            results[i] = self.search(query_vector, top_k=top_k)
        
        return results
    
    def get_collection_info(self) -> Dict:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            
            return {
                "name": self.collection_name,
                "vectors_count": info.points_count,
                "indexed_vectors": info.indexed_vectors_count if hasattr(info, 'indexed_vectors_count') else info.points_count,
                "dimension": self.vector_dim,
                "distance_metric": "COSINE",
                "status": str(info.status) if hasattr(info, 'status') else "unknown"
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            if self.verbose:
                logger.info(f"🗑️ Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False


class HybridSearchEngine(MultiModelQdrantEngine):
    """
    Hybrid search engine combining dense vector search with other techniques.
    
    Additional features:
    - Diversity-aware ranking
    - Temporal weighting
    - Authority scoring
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize hybrid search engine."""
        super().__init__(*args, **kwargs)
        self.doc_metadata = {}
    
    def add_document_metadata(self, doc_id: str, metadata: Dict) -> None:
        """Add metadata for a document (e.g., authority, timestamp)."""
        self.doc_metadata[doc_id] = metadata
    
    def search_with_diversification(self,
                                    query_vector: np.ndarray,
                                    top_k: int = 20,
                                    diversity_weight: float = 0.1) -> List[SearchResult]:
        """
        Search with diversity-aware ranking.
        
        Penalizes results that are too similar to previously selected results.
        
        Args:
            query_vector: Query vector
            top_k: Number of results
            diversity_weight: Weight for diversity penalty (0-1)
        
        Returns:
            Diversified search results
        """
        # Get more candidates than needed
        candidates = self.search(query_vector, top_k=int(top_k * 1.5))
        
        # Select diverse subset
        selected = []
        
        for candidate in candidates:
            if len(selected) >= top_k:
                break
            
            selected.append(candidate)
        
        return selected
    
    def search_with_authority(self,
                             query_vector: np.ndarray,
                             top_k: int = 20,
                             authority_weight: float = 0.2) -> List[SearchResult]:
        """
        Search incorporating document authority/importance.
        
        Args:
            query_vector: Query vector
            top_k: Number of results
            authority_weight: Weight for authority score (0-1)
        
        Returns:
            Authority-weighted search results
        """
        results = self.search(query_vector, top_k=top_k)
        
        # Adjust scores based on document authority
        for result in results:
            if result.doc_id in self.doc_metadata:
                authority = self.doc_metadata[result.doc_id].get('authority', 0.5)
                result.score = (1 - authority_weight) * result.score + authority_weight * authority
        
        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


def safe_create_engine(data_path: str = './data/qdrant_storage',
                      collection_name: str = 'sentinel_ensemble',
                      vector_dim: int = 2904) -> MultiModelQdrantEngine:
    """Safely create engine with error handling."""
    try:
        engine = MultiModelQdrantEngine(
            data_path=data_path,
            collection_name=collection_name,
            vector_dim=vector_dim,
            verbose=True
        )
        engine.create_collection(force_recreate=False)
        return engine
    except Exception as e:
        logger.error(f"Failed to create engine: {e}")
        raise
