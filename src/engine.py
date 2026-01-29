from qdrant_client import QdrantClient, models
from .config import DATA_PATH, COLLECTION_NAME, VECTOR_DIM, GT_COLLECTION, BQ_COLLECTION

class SentinelEngine:
    def __init__(self, data_path=DATA_PATH, collection_name=COLLECTION_NAME, vector_dim=VECTOR_DIM, verbose=False):
        self.client = QdrantClient(path=data_path)
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.verbose = verbose

    def init_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True # Essential for 100K scale
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                )
            )

    def init_collections(self):
        if not self.client.collection_exists(GT_COLLECTION):
            self.client.create_collection(
                collection_name=GT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
            )
        if not self.client.collection_exists(BQ_COLLECTION):
            self.client.create_collection(
                collection_name=BQ_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                ),
            )

    def upsert_vectors(self, vectors, point_ids, batch_size=128, collection_name=None, payloads=None):
        target_collection = collection_name or self.collection_name
        for i in range(0, len(vectors), batch_size):
            batch_end = min(i + batch_size, len(vectors))
            batch_ids = point_ids[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_payloads = None
            if payloads:
                batch_payloads = payloads[i:batch_end]
            self.client.upsert(
                collection_name=target_collection,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=batch_vectors.tolist(),
                    payloads=batch_payloads,
                ),
            )

    def ingest(self, vectors, doc_texts, ids=None, batch_size=128):
        if ids is None:
            ids = list(range(len(vectors)))
        payloads = [{"text": text} for text in doc_texts]
        self.upsert_vectors(vectors, ids, batch_size=batch_size, collection_name=GT_COLLECTION, payloads=payloads)
        self.upsert_vectors(vectors, ids, batch_size=batch_size, collection_name=BQ_COLLECTION, payloads=payloads)

    def search(self, vector, top_k=10):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
        )
        return [(str(point.id), point.score) for point in response.points]

    def confidence_driven_search(self, vector, k=10, oversample=4.0):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=int(k * oversample),
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=oversample,
                )
            ),
        )
        return response.points[:k]

    def sovereign_search(self, vector, oversample=4.0, k=10):
        response = self.client.query_points(
            collection_name=BQ_COLLECTION,
            query=vector,
            limit=int(k * oversample),
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=oversample,
                )
            ),
        )
        return response

    def get_collection_info(self):
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
        }

    def close(self):
        """Explicitly close the client to prevent __del__ exceptions"""
        self.client.close()
