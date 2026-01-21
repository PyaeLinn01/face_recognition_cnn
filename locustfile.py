"""
Locust load testing for Face Attendance Streamlit App.

This file tests the Streamlit application's HTTP endpoints.
Streamlit uses WebSocket for real-time updates, but we can still
load test the initial page load and API endpoints.

Usage:
    # Start the Streamlit app first:
    streamlit run attend_app.py --server.port 8501

    # Then run Locust:
    locust -f locustfile.py --host=http://localhost:8501

    # Open browser to http://localhost:8089 for Locust UI
    # Set number of users, spawn rate, and start swarming

    # Or run headless:
    locust -f locustfile.py --host=http://localhost:8501 --users 10 --spawn-rate 2 --run-time 60s --headless
"""

import time
import random
from locust import HttpUser, task, between, events


class StreamlitUser(HttpUser):
    """
    Simulates users accessing the Face Attendance Streamlit app.
    
    Streamlit apps have specific endpoints:
    - / : Main page (HTML)
    - /_stcore/health : Health check endpoint
    - /_stcore/stream : WebSocket for updates (not tested here)
    - /static/* : Static assets
    """
    
    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Initial page load to establish session
        self.client.get("/", name="/ (initial load)")
    
    @task(10)
    def load_main_page(self):
        """
        Load the main Streamlit page.
        This is the most common user action.
        """
        with self.client.get("/", catch_response=True, name="/ (main page)") as response:
            if response.status_code == 200:
                if "streamlit" in response.text.lower() or "face" in response.text.lower():
                    response.success()
                else:
                    response.failure("Page loaded but content unexpected")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(5)
    def health_check(self):
        """
        Check Streamlit health endpoint.
        Used by load balancers and monitoring.
        """
        with self.client.get("/_stcore/health", catch_response=True, name="/_stcore/health") as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Older Streamlit versions may not have this endpoint
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(3)
    def load_static_assets(self):
        """
        Simulate loading static assets that the browser would request.
        """
        static_paths = [
            "/_stcore/host-config",
            "/_stcore/allowed-message-origins",
        ]
        
        path = random.choice(static_paths)
        with self.client.get(path, catch_response=True, name=f"{path}") as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(2)
    def simulate_page_interaction(self):
        """
        Simulate a user spending time on the page.
        """
        # Load main page
        self.client.get("/", name="/ (interaction)")
        
        # Simulate user reading/interacting
        time.sleep(random.uniform(0.5, 2.0))
        
        # Reload (simulating widget interaction which causes rerun)
        self.client.get("/", name="/ (rerun)")


# Event hooks for custom logging
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Log slow requests for debugging."""
    if response_time > 5000:  # More than 5 seconds
        print(f"SLOW REQUEST: {name} took {response_time}ms")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 60)
    print("Starting Face Attendance Load Test")
    print("=" * 60)
    print(f"Target host: {environment.host}")
    print("Make sure Streamlit app is running!")
    print("=" * 60)


@events.test_stop.add_listener  
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("=" * 60)
    print("Load Test Complete")
    print("=" * 60)


# For running directly with: python locustfile.py
if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py --host=http://localhost:8501")
