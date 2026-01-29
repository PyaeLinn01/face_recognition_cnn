"""
Locust load testing for Face Attendance Frontend (Vite) and Backend API.

Usage:
    # Frontend (Vite) is on http://localhost:8080
    # Backend API is on http://localhost:5001

    locust -f locustfile.py

Environment variables:
    FRONTEND_HOST=http://localhost:8080
    BACKEND_HOST=http://localhost:5001
    ADMIN_EMAIL=admin@gmail.com
    ADMIN_PASSWORD=123456
    TEACHER_EMAIL=teacher@gmail.com
    TEACHER_PASSWORD=123456
"""

import os
import re
import time
import random
from typing import List, Optional
from locust import HttpUser, task, between, events


FRONTEND_HOST = os.getenv("FRONTEND_HOST", "http://localhost:8080")
BACKEND_HOST = os.getenv("BACKEND_HOST", "http://localhost:5001")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@gmail.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "123456")
TEACHER_EMAIL = os.getenv("TEACHER_EMAIL", "teacher@gmail.com")
TEACHER_PASSWORD = os.getenv("TEACHER_PASSWORD", "123456")


def _extract_asset_paths(html: str) -> List[str]:
    """Extract /assets/* paths from HTML (Vite build)."""
    return list(set(re.findall(r"/assets/[^\"'\s>]+", html)))


class FrontendUser(HttpUser):
    """Simulates users accessing the React frontend."""

    host = FRONTEND_HOST
    wait_time = between(1, 3)

    def on_start(self):
        self.asset_paths: List[str] = []
        with self.client.get("/", catch_response=True, name="/ (initial load)") as response:
            if response.status_code == 200:
                self.asset_paths = _extract_asset_paths(response.text)
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(10)
    def load_home(self):
        with self.client.get("/", catch_response=True, name="/ (home)") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def load_assets(self):
        if not self.asset_paths:
            self.client.get("/favicon.ico", name="/favicon.ico")
            return
        asset = random.choice(self.asset_paths)
        self.client.get(asset, name="/assets/*")

    @task(2)
    def simulate_user_wait(self):
        time.sleep(random.uniform(0.5, 2.0))


class BackendUser(HttpUser):
    """Simulates users calling backend API endpoints."""

    host = BACKEND_HOST
    wait_time = between(1, 3)

    def on_start(self):
        self.user_id: Optional[str] = None
        self._login_admin()

    def _login_admin(self):
        payload = {"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
        with self.client.post("/api/v1/auth/login", json=payload, name="POST /api/v1/auth/login") as response:
            if response.status_code == 200:
                data = response.json()
                self.user_id = data.get("user", {}).get("id")
                response.success()
            else:
                response.failure(f"Login failed: {response.status_code}")

    @task(8)
    def health_check(self):
        self.client.get("/health", name="GET /health")

    @task(6)
    def list_majors(self):
        self.client.get("/api/v1/admin/majors", name="GET /api/v1/admin/majors")

    @task(6)
    def list_subjects(self):
        self.client.get("/api/v1/admin/subjects", name="GET /api/v1/admin/subjects")

    @task(4)
    def list_teachers(self):
        self.client.get("/api/v1/admin/teachers", name="GET /api/v1/admin/teachers")

    @task(4)
    def list_students(self):
        self.client.get("/api/v1/admin/students", name="GET /api/v1/admin/students")

    @task(4)
    def attendance_recent(self):
        self.client.get("/api/v1/attendance/recent", name="GET /api/v1/attendance/recent")

    @task(3)
    def attendance_stats(self):
        self.client.get("/api/v1/teacher/attendance/stats", name="GET /api/v1/teacher/attendance/stats")

    @task(2)
    def list_faces(self):
        self.client.get("/api/v1/faces/list", name="GET /api/v1/faces/list")

    @task(2)
    def get_user_profile(self):
        if not self.user_id:
            return
        self.client.get(f"/api/v1/auth/user/{self.user_id}", name="GET /api/v1/auth/user/:id")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    if response_time > 5000:
        print(f"SLOW REQUEST: {name} took {response_time}ms")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("=" * 60)
    print("Starting Face Attendance Load Test")
    print("=" * 60)
    print(f"Frontend host: {FRONTEND_HOST}")
    print(f"Backend host: {BACKEND_HOST}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("=" * 60)
    print("Load Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    os.system("locust -f locustfile.py")
