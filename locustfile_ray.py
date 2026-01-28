"""
Ray-optimized Locust load testing for distributed Face Recognition System.

This file tests the Ray-based distributed face recognition system with:
- Multiple concurrent users
- Batch processing capabilities
- Distributed model inference
- Database operation scaling
"""

import time
import random
import base64
import json
from typing import Dict, List, Optional
from locust import HttpUser, task, between, events, FastHttpUser
import requests


class RayFaceRecognitionUser(FastHttpUser):
    """
    Simulates users interacting with Ray-based face recognition API.

    Tests distributed endpoints for:
    - Face recognition (single and batch)
    - Face registration
    - Performance under concurrent load
    """

    # Wait between 0.5 and 2 seconds between tasks
    wait_time = between(0.5, 2)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_images = []
        self.registered_identities = []

    def on_start(self):
        """Called when a simulated user starts."""
        # Pre-load test images (dummy data for simulation)
        self.test_images = self._generate_test_images(10)
        self.registered_identities = [f"user_{i}" for i in range(10)]

        # Register a few test identities at startup
        for i in range(3):
            self._register_test_identity(f"test_user_{self.__class__.__name__}_{i}")

    def _generate_test_images(self, count: int) -> List[str]:
        """
        Generate dummy base64-encoded images for testing.

        In a real scenario, these would be actual face images.
        """
        images = []
        # Create dummy 96x96 RGB images (compressed as base64)
        dummy_image_data = "iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAYAAADimHc4AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAFYSURBVHic7d1BaxRBFIXh"  # Truncated for brevity

        for _ in range(count):
            images.append(dummy_image_data)
        return images

    def _register_test_identity(self, identity: str) -> None:
        """Register a test identity with the system."""
        try:
            image_hex = random.choice(self.test_images)

            payload = {
                "identity": identity,
                "image_hex": image_hex,
                "use_detection": True,
                "min_confidence": 0.90
            }

            with self.client.post(
                "/register",
                json=payload,
                catch_response=True,
                name="/register (setup)"
            ) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Registration failed: {response.status_code}")

        except Exception as e:
            self.environment.events.request.fire(
                request_type="POST",
                name="/register (setup)",
                response_time=0,
                response_length=0,
                response=None,
                context={},
                exception=e,
            )

    @task(5)
    def recognize_single_face(self):
        """
        Test single face recognition endpoint.

        This is the most common operation - recognizing one face at a time.
        """
        image_hex = random.choice(self.test_images)
        threshold = random.uniform(0.5, 0.9)

        payload = {
            "image_hex": image_hex,
            "threshold": threshold,
            "use_detection": True,
            "min_confidence": 0.85
        }

        with self.client.post(
            "/recognize",
            json=payload,
            catch_response=True,
            name="/recognize (single)"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "recognized" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 429:
                # Rate limited - this is expected under high load
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def recognize_batch_faces(self):
        """
        Test batch face recognition endpoint.

        Tests the system's ability to process multiple faces concurrently.
        """
        batch_size = random.randint(2, 5)
        image_batch = random.sample(self.test_images, min(batch_size, len(self.test_images)))

        payload = {
            "image_batch": image_batch,
            "threshold": 0.7,
            "use_detection": True,
            "min_confidence": 0.85
        }

        with self.client.post(
            "/batch_recognize",
            json=payload,
            catch_response=True,
            name=f"/batch_recognize ({batch_size} faces)"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list) and len(data) == batch_size:
                        response.success()
                    else:
                        response.failure("Invalid batch response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def register_new_face(self):
        """
        Test face registration endpoint.

        Simulates new users registering their faces.
        """
        identity = f"new_user_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        image_hex = random.choice(self.test_images)

        payload = {
            "identity": identity,
            "image_hex": image_hex,
            "use_detection": True,
            "min_confidence": 0.90
        }

        with self.client.post(
            "/register",
            json=payload,
            catch_response=True,
            name="/register (new user)"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("success", False):
                        response.success()
                        # Add to registered identities for future recognition tests
                        self.registered_identities.append(identity)
                    else:
                        response.failure("Registration not successful")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """
        Check system health and performance metrics.

        Tests monitoring endpoints and system status.
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def get_system_metrics(self):
        """
        Retrieve system performance metrics.

        Tests monitoring and observability endpoints.
        """
        with self.client.get(
            "/metrics",
            catch_response=True,
            name="/metrics"
        ) as response:
            if response.status_code in [200, 404]:  # 404 if metrics not implemented
                response.success()
            else:
                response.failure(f"Metrics retrieval failed: {response.status_code}")


class HighLoadRayUser(RayFaceRecognitionUser):
    """
    High-load user class for stress testing.

    Focuses on maximum throughput with minimal wait times.
    """
    wait_time = between(0.1, 0.5)

    @task(8)
    def stress_recognize(self):
        """High-frequency recognition requests."""
        self.recognize_single_face()

    @task(2)
    def stress_batch(self):
        """High-frequency batch processing."""
        self.recognize_batch_faces()


# Custom event handlers for detailed logging
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response,
               context, exception, **kwargs):
    """Log slow requests for performance analysis."""
    if response_time > 2000:  # More than 2 seconds
        print(f"SLOW REQUEST: {name} took {response_time}ms")

    # Log errors for debugging
    if exception or (response and response.status_code >= 400):
        print(f"ERROR in {name}: {exception or response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 70)
    print("Starting Ray-based Face Recognition Load Test")
    print("=" * 70)
    print("Testing distributed face recognition with Ray")
    print(f"Target host: {environment.host}")
    print("Ensure Ray cluster and API server are running!")
    print("=" * 70)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("=" * 70)
    print("Ray Face Recognition Load Test Complete")
    print("=" * 70)
    print("Performance metrics:")
    print("- Check response times for recognition vs batch operations")
    print("- Monitor Ray dashboard for actor utilization")
    print("- Review error rates and throughput")
    print("=" * 70)


# Performance tracking for Ray-specific metrics
class RayMetricsTracker:
    """Track Ray-specific performance metrics during load testing."""

    def __init__(self):
        self.metrics = {
            "recognition_requests": 0,
            "batch_requests": 0,
            "registration_requests": 0,
            "errors": 0,
            "avg_response_time": 0,
            "throughput": 0
        }

    def track_request(self, name: str, response_time: float, success: bool):
        """Track a request for metrics."""
        if "recognize" in name and "batch" not in name:
            self.metrics["recognition_requests"] += 1
        elif "batch" in name:
            self.metrics["batch_requests"] += 1
        elif "register" in name:
            self.metrics["registration_requests"] += 1

        if not success:
            self.metrics["errors"] += 1

        # Update rolling average response time
        total_requests = sum([
            self.metrics["recognition_requests"],
            self.metrics["batch_requests"],
            self.metrics["registration_requests"]
        ])

        if total_requests > 0:
            self.metrics["avg_response_time"] = (
                (self.metrics["avg_response_time"] * (total_requests - 1)) + response_time
            ) / total_requests


# Global metrics tracker
metrics_tracker = RayMetricsTracker()

@events.request.add_listener
def track_metrics(request_type, name, response_time, response_length, response,
                 context, exception, **kwargs):
    """Track custom metrics for Ray performance analysis."""
    success = exception is None and (response is None or response.status_code < 400)
    metrics_tracker.track_request(name, response_time, success)


if __name__ == "__main__":
    import os
    # Run with Ray API server
    ray_host = os.getenv("RAY_API_HOST", "http://localhost:8000")
    os.system(f"locust -f locustfile_ray.py --host={ray_host} --users 50 --spawn-rate 10")