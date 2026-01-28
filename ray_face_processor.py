"""
Ray-based distributed face processing for Face Recognition CNN.

This module provides Ray Actors and Tasks for scalable face recognition operations:
- Distributed face embedding computation
- Batch processing of multiple faces
- Cached model inference
- Database operations with connection pooling
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import ray
from ray import serve
import cv2
import tensorflow as tf
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import mlflow

# Import existing utilities
from app import (
    get_facenet_model,
    embedding_from_bytes,
    _preprocess_bgr_image,
    _embedding_from_preprocessed
)
from mtcnn.mtcnn import MTCNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0.5, num_gpus=0.25)
class FaceEmbeddingActor:
    """
    Ray Actor for cached FaceNet model inference.

    Maintains a cached TensorFlow model instance to avoid
    reloading weights for each inference request.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the FaceNet model actor."""
        self.model = None
        self.mtcnn_detector = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load and cache the FaceNet model."""
        try:
            logger.info("Loading FaceNet model in Ray actor...")
            self.model = get_facenet_model()

            # Load custom weights if specified
            if self.model_path and Path(self.model_path).exists():
                self.model.load_weights(self.model_path)
                logger.info(f"Loaded custom weights from {self.model_path}")

            # Initialize MTCNN detector
            self.mtcnn_detector = MTCNN()

            logger.info("FaceNet model and MTCNN detector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @ray.method(num_returns=1)
    def compute_embedding(
        self,
        image_bytes: bytes,
        use_detection: bool = True,
        require_detection: bool = False,
        min_confidence: float = 0.90,
        margin_ratio: float = 0.20,
        use_alignment: bool = True,
        use_prewhitening: bool = False,
        use_tta_flip: bool = True,
    ) -> np.ndarray:
        """
        Compute face embedding from image bytes.

        Args:
            image_bytes: Raw image bytes
            use_detection: Whether to use MTCNN face detection
            require_detection: Whether to require face detection
            min_confidence: Minimum face detection confidence
            margin_ratio: Face crop margin ratio
            use_alignment: Whether to align faces by keypoints
            use_prewhitening: Whether to apply prewhitening
            use_tta_flip: Whether to use test-time augmentation with flipping

        Returns:
            Face embedding vector as numpy array
        """
        try:
            # Process image bytes to embedding
            embedding, _ = embedding_from_bytes(
                image_bytes=image_bytes,
                model=self.model,
                use_detection=use_detection,
                require_detection=require_detection,
                min_confidence=min_confidence,
                margin_ratio=margin_ratio,
                use_alignment=use_alignment,
                use_prewhitening=use_prewhitening,
                use_tta_flip=use_tta_flip,
            )
            return embedding
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            raise

    @ray.method(num_returns=1)
    def batch_compute_embeddings(
        self,
        image_batch: List[bytes],
        **kwargs
    ) -> List[np.ndarray]:
        """
        Compute embeddings for a batch of images.

        Args:
            image_batch: List of image bytes
            **kwargs: Embedding computation parameters

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for img_bytes in image_batch:
            try:
                emb = self.compute_embedding(img_bytes, **kwargs)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Failed to process image in batch: {e}")
                # Return zero embedding for failed images
                embeddings.append(np.zeros((1, 128), dtype=np.float32))

        return embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "trainable_params": sum([layer.count_params() for layer in self.model.trainable_weights]),
            "non_trainable_params": sum([layer.count_params() for layer in self.model.non_trainable_weights]),
        }


@ray.remote(num_cpus=0.2)
class DatabaseActor:
    """
    Ray Actor for database operations with connection pooling.

    Handles MongoDB operations for face storage and attendance tracking.
    """

    def __init__(self, connection_string: str, database_name: str):
        """Initialize database connection."""
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        self.connect()

    def connect(self):
        """Establish database connection."""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    @ray.method(num_returns=1)
    def save_face_embedding(
        self,
        identity: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save face embedding to database.

        Args:
            identity: Person identity/name
            embedding: Face embedding vector
            metadata: Additional metadata

        Returns:
            Success status
        """
        try:
            collection = self.db.face_embeddings

            doc = {
                "identity": identity,
                "embedding": embedding.flatten().tolist(),
                "timestamp": time.time(),
                "metadata": metadata or {}
            }

            result = collection.insert_one(doc)
            logger.info(f"Saved embedding for {identity}, ID: {result.inserted_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save embedding: {e}")
            return False

    @ray.method(num_returns=1)
    def find_similar_embeddings(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar face embeddings using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            threshold: Similarity threshold
            limit: Maximum number of results

        Returns:
            List of similar embeddings with metadata
        """
        try:
            collection = self.db.face_embeddings

            # Get all embeddings (in production, use vector index)
            all_docs = list(collection.find({}, {"identity": 1, "embedding": 1, "_id": 0}))

            results = []
            query_emb = query_embedding.flatten()

            for doc in all_docs:
                stored_emb = np.array(doc["embedding"])
                # Cosine similarity
                similarity = np.dot(query_emb, stored_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(stored_emb)
                )
                distance = 1.0 - similarity

                if distance < threshold:
                    results.append({
                        "identity": doc["identity"],
                        "distance": float(distance),
                        "similarity": float(similarity)
                    })

            # Sort by distance (ascending)
            results.sort(key=lambda x: x["distance"])
            return results[:limit]

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    @ray.method(num_returns=1)
    def record_attendance(
        self,
        identity: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record attendance event.

        Args:
            identity: Recognized identity
            confidence: Recognition confidence score
            metadata: Additional attendance metadata

        Returns:
            Success status
        """
        try:
            collection = self.db.attendance

            doc = {
                "identity": identity,
                "confidence": float(confidence),
                "timestamp": time.time(),
                "metadata": metadata or {}
            }

            collection.insert_one(doc)
            logger.info(f"Recorded attendance for {identity}")
            return True
        except Exception as e:
            logger.error(f"Failed to record attendance: {e}")
            return False


@ray.remote
def batch_process_faces(
    image_batch: List[bytes],
    embedding_actor: ray.actor.ActorHandle,
    **embedding_kwargs
) -> List[np.ndarray]:
    """
    Ray Task for batch processing multiple face images.

    Distributes face processing across available actors.

    Args:
        image_batch: List of image bytes
        embedding_actor: Handle to FaceEmbeddingActor
        **embedding_kwargs: Embedding computation parameters

    Returns:
        List of computed embeddings
    """
    try:
        # Process batch using the actor
        embeddings = ray.get(
            embedding_actor.batch_compute_embeddings.remote(
                image_batch, **embedding_kwargs
            )
        )
        return embeddings
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return []


@ray.remote
def distributed_similarity_search(
    query_embedding: np.ndarray,
    database_actor: ray.actor.ActorHandle,
    threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Distributed similarity search task.

    Args:
        query_embedding: Query embedding vector
        database_actor: Handle to DatabaseActor
        threshold: Similarity threshold

    Returns:
        Search results
    """
    try:
        results = ray.get(
            database_actor.find_similar_embeddings.remote(
                query_embedding, threshold
            )
        )
        return results
    except Exception as e:
        logger.error(f"Distributed search failed: {e}")
        return []


class FaceRecognitionService:
    """
    High-level service coordinating Ray-based face recognition operations.
    """

    def __init__(
        self,
        num_embedding_actors: int = 2,
        num_db_actors: int = 1,
        mongo_connection: str = "mongodb://localhost:27017",
        mongo_db: str = "face_recognition"
    ):
        """Initialize the face recognition service."""
        self.embedding_actors = []
        self.db_actors = []

        # Start Ray if not already running
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create embedding actors
        for i in range(num_embedding_actors):
            actor = FaceEmbeddingActor.remote()
            self.embedding_actors.append(actor)

        # Create database actors
        for i in range(num_db_actors):
            actor = DatabaseActor.remote(mongo_connection, mongo_db)
            self.db_actors.append(actor)

        logger.info(f"Started {num_embedding_actors} embedding actors and {num_db_actors} DB actors")

    def recognize_face(
        self,
        image_bytes: bytes,
        threshold: float = 0.7,
        **embedding_kwargs
    ) -> Dict[str, Any]:
        """
        Recognize a face from image bytes.

        Args:
            image_bytes: Raw image bytes
            threshold: Recognition threshold
            **embedding_kwargs: Embedding computation parameters

        Returns:
            Recognition result
        """
        try:
            # Get embedding from first available actor
            embedding_future = self.embedding_actors[0].compute_embedding.remote(
                image_bytes, **embedding_kwargs
            )

            # Search for similar faces
            search_future = self.db_actors[0].find_similar_embeddings.remote(
                ray.get(embedding_future), threshold
            )

            # Wait for results
            search_results = ray.get(search_future)

            if search_results:
                best_match = search_results[0]
                return {
                    "recognized": True,
                    "identity": best_match["identity"],
                    "confidence": 1.0 - best_match["distance"],
                    "distance": best_match["distance"]
                }
            else:
                return {
                    "recognized": False,
                    "identity": None,
                    "confidence": 0.0,
                    "distance": float('inf')
                }

        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return {
                "recognized": False,
                "identity": None,
                "confidence": 0.0,
                "error": str(e)
            }

    def register_face(
        self,
        identity: str,
        image_bytes: bytes,
        **embedding_kwargs
    ) -> bool:
        """
        Register a new face in the database.

        Args:
            identity: Person identity
            image_bytes: Face image bytes
            **embedding_kwargs: Embedding computation parameters

        Returns:
            Registration success status
        """
        try:
            # Compute embedding
            embedding_future = self.embedding_actors[0].compute_embedding.remote(
                image_bytes, **embedding_kwargs
            )
            embedding = ray.get(embedding_future)

            # Save to database
            save_future = self.db_actors[0].save_face_embedding.remote(
                identity, embedding
            )

            success = ray.get(save_future)
            return success

        except Exception as e:
            logger.error(f"Face registration failed: {e}")
            return False

    def batch_recognize(
        self,
        image_batch: List[bytes],
        threshold: float = 0.7,
        **embedding_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch recognize multiple faces.

        Args:
            image_batch: List of image bytes
            threshold: Recognition threshold
            **embedding_kwargs: Embedding computation parameters

        Returns:
            List of recognition results
        """
        try:
            # Distribute batch processing across actors
            batch_size = len(image_batch) // len(self.embedding_actors) + 1
            futures = []

            for i in range(0, len(image_batch), batch_size):
                batch = image_batch[i:i + batch_size]
                actor = self.embedding_actors[i // batch_size % len(self.embedding_actors)]

                future = batch_process_faces.remote(
                    batch, actor, **embedding_kwargs
                )
                futures.append(future)

            # Collect embeddings
            all_embeddings = []
            for future in futures:
                embeddings = ray.get(future)
                all_embeddings.extend(embeddings)

            # Perform similarity search for each embedding
            recognition_futures = []
            for embedding in all_embeddings:
                future = distributed_similarity_search.remote(
                    embedding, self.db_actors[0], threshold
                )
                recognition_futures.append(future)

            # Collect results
            results = []
            for future in recognition_futures:
                search_results = ray.get(future)
                if search_results:
                    best_match = search_results[0]
                    results.append({
                        "recognized": True,
                        "identity": best_match["identity"],
                        "confidence": 1.0 - best_match["distance"],
                        "distance": best_match["distance"]
                    })
                else:
                    results.append({
                        "recognized": False,
                        "identity": None,
                        "confidence": 0.0,
                        "distance": float('inf')
                    })

            return results

        except Exception as e:
            logger.error(f"Batch recognition failed: {e}")
            return []

    def shutdown(self):
        """Shutdown the service and cleanup actors."""
        try:
            # Kill all actors
            for actor in self.embedding_actors + self.db_actors:
                ray.kill(actor)

            logger.info("Face recognition service shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Ray Serve deployment for HTTP API
@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1, "num_gpus": 0.5})
class FaceRecognitionDeployment:
    """Ray Serve deployment for face recognition API."""

    def __init__(self):
        """Initialize the deployment."""
        self.service = FaceRecognitionService(
            num_embedding_actors=2,
            num_db_actors=1
        )

    async def recognize(self, request):
        """Handle face recognition requests."""
        try:
            data = await request.json()
            image_bytes = bytes.fromhex(data["image_hex"])
            threshold = data.get("threshold", 0.7)

            result = self.service.recognize_face(image_bytes, threshold)
            return result
        except Exception as e:
            return {"error": str(e), "recognized": False}

    async def register(self, request):
        """Handle face registration requests."""
        try:
            data = await request.json()
            identity = data["identity"]
            image_bytes = bytes.fromhex(data["image_hex"])

            success = self.service.register_face(identity, image_bytes)
            return {"success": success}
        except Exception as e:
            return {"success": False, "error": str(e)}


# FastAPI-style route definitions for Ray Serve
face_recognition_app = FaceRecognitionDeployment.bind()


if __name__ == "__main__":
    # Example usage
    service = FaceRecognitionService()

    # Test with dummy data
    print("Face Recognition Service initialized with Ray")

    # Cleanup
    service.shutdown()