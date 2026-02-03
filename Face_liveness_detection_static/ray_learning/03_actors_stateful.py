"""
==============================================================================
LESSON 3: ACTORS - STATEFUL DISTRIBUTED COMPUTATION
==============================================================================

🎯 Learning Objectives:
    1. Understand when to use Actors vs Tasks
    2. Create actors with @ray.remote on classes
    3. Call actor methods and manage state
    4. Pass actor handles between tasks
    5. Named actors and detached actors
    6. Actor pools for scalable stateful workloads

⏱️ Estimated time: 35-45 minutes

Prerequisites: Complete lessons 01 and 02 first

Run this file: python 03_actors_stateful.py
==============================================================================
"""

import ray
import time
import random
import os

ray.init(ignore_reinit_error=True)

# ============================================================================
# SECTION 1: TASKS VS ACTORS - WHEN TO USE WHICH?
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TASKS vs ACTORS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ TASKS (@ray.remote on functions)    │ ACTORS (@ray.remote on classes)      │
├─────────────────────────────────────┼───────────────────────────────────────┤
│ ✓ Stateless                         │ ✓ Stateful                            │
│ ✓ Each call is independent          │ ✓ State persists between calls       │
│ ✓ Can run on any worker             │ ✓ Runs on dedicated worker           │
│ ✓ Embarrassingly parallel           │ ✓ Sequential method execution        │
│                                     │                                       │
│ Use for:                            │ Use for:                              │
│ - Pure functions                    │ - ML model inference (model in RAM)  │
│ - Data transformation               │ - Database connections                │
│ - One-off computations              │ - Caching/memoization                │
│ - Batch processing                  │ - Simulations with state             │
│                                     │ - Rate limiters, counters            │
└─────────────────────────────────────┴───────────────────────────────────────┘

Key insight: Actors are WORKERS with persistent state.
Each actor instance runs in its own dedicated process.
"""

print("=" * 60)
print("SECTION 1: Tasks vs Actors Comparison")
print("=" * 60)

# PROBLEM: Counting with TASKS (doesn't work!)
@ray.remote
def increment_task(counter):
    """Try to increment a counter with tasks - THIS WON'T WORK!"""
    counter["value"] += 1
    return counter["value"]

print("\n❌ Trying to share state with TASKS:")
shared_counter = {"value": 0}
futures = [increment_task.remote(shared_counter) for _ in range(5)]
results = ray.get(futures)
print(f"   Results: {results}")
print(f"   Expected [1, 2, 3, 4, 5] but got all 1s!")
print("   (Each task gets a COPY of the counter - no shared state!)")

# SOLUTION: Counting with ACTORS (works!)
@ray.remote
class Counter:
    """Actor that maintains a counter."""
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value
    
    def get_value(self):
        return self.value

print("\n✅ Sharing state with ACTORS:")
counter_actor = Counter.remote()  # Create the actor
futures = [counter_actor.increment.remote() for _ in range(5)]
results = ray.get(futures)
print(f"   Results: {results}")
print("   Each call sees the updated state from previous calls!")


# ============================================================================
# SECTION 2: CREATING AND USING ACTORS
# ============================================================================
"""
Actor lifecycle:
1. Define class with @ray.remote
2. Create instance with ClassName.remote(*args)
3. Call methods with actor.method.remote(*args)
4. Get results with ray.get()

Important: 
- Constructor runs on a remote worker
- Methods execute SEQUENTIALLY on that worker (thread-safe!)
- Actor lives until explicitly killed or driver exits
"""

print("\n" + "=" * 60)
print("SECTION 2: Creating and Using Actors")
print("=" * 60)

@ray.remote
class MLModel:
    """Simulates a ML model that's expensive to load."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.load_time = time.time()
        print(f"   [Actor] Loading model '{model_name}' (PID: {os.getpid()})")
        time.sleep(1)  # Simulate slow model loading
        self.inference_count = 0
        print(f"   [Actor] Model '{model_name}' ready!")
    
    def predict(self, input_data):
        """Run inference."""
        self.inference_count += 1
        time.sleep(0.1)  # Simulate inference
        return {
            "model": self.model_name,
            "input": input_data,
            "prediction": input_data * 2,  # Silly prediction
            "inference_num": self.inference_count
        }
    
    def get_stats(self):
        """Return model statistics."""
        return {
            "model_name": self.model_name,
            "total_inferences": self.inference_count,
            "uptime": time.time() - self.load_time
        }

print("\n🤖 Creating ML model actor...")
model = MLModel.remote("resnet50")  # Constructor runs remotely!

# Give it a moment to load
time.sleep(1.5)

# Run some predictions
print("\n📊 Running predictions:")
for i in range(3):
    result = ray.get(model.predict.remote(i + 1))
    print(f"   Prediction {result['inference_num']}: {result['input']} → {result['prediction']}")

# Get statistics
stats = ray.get(model.get_stats.remote())
print(f"\n📈 Model stats: {stats}")


# ============================================================================
# SECTION 3: ACTOR METHODS EXECUTE SEQUENTIALLY
# ============================================================================
"""
Critical concept: All method calls on a single actor execute SEQUENTIALLY.

This is by design - it means:
✅ No race conditions on actor state
✅ No need for locks inside actors
✅ Predictable state transitions

BUT: This can be a bottleneck!
Solution: Create multiple actor instances (actor pool)
"""

print("\n" + "=" * 60)
print("SECTION 3: Sequential Method Execution")
print("=" * 60)

@ray.remote
class SequentialDemo:
    """Demonstrates sequential execution."""
    
    def __init__(self):
        self.call_order = []
    
    def slow_method(self, call_id):
        """Takes 0.5s to execute."""
        self.call_order.append(f"start_{call_id}")
        time.sleep(0.5)
        self.call_order.append(f"end_{call_id}")
        return call_id
    
    def get_order(self):
        return self.call_order

print("\n🔄 Calling method 4 times on SAME actor...")
demo = SequentialDemo.remote()

start = time.time()
# Submit all calls at once (async)
futures = [demo.slow_method.remote(i) for i in range(4)]
# Wait for all to complete
results = ray.get(futures)
print(f"   Results: {results}")
print(f"   Time: {time.time() - start:.2f}s (sequential: 4 × 0.5s = 2s)")

order = ray.get(demo.get_order.remote())
print(f"   Execution order: {order}")
print("   (Notice: start_0, end_0, start_1, end_1... - strictly sequential!)")


# ============================================================================
# SECTION 4: MULTIPLE ACTORS FOR PARALLELISM
# ============================================================================
"""
To get parallelism with actors, create multiple instances!

This is the Actor Pool pattern:
1. Create N actor instances
2. Distribute work across them
3. Each actor processes requests sequentially
4. Across actors, work happens in parallel
"""

print("\n" + "=" * 60)
print("SECTION 4: Multiple Actors for Parallelism")
print("=" * 60)

@ray.remote
class Worker:
    """A worker that processes items."""
    
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.items_processed = 0
    
    def process(self, item):
        """Process an item."""
        time.sleep(0.3)
        self.items_processed += 1
        return f"Worker {self.worker_id} processed item {item}"
    
    def get_count(self):
        return self.items_processed

# Create a pool of 4 workers
NUM_WORKERS = 4
print(f"\n👷 Creating pool of {NUM_WORKERS} workers...")
workers = [Worker.remote(i) for i in range(NUM_WORKERS)]

# Distribute 12 items across workers (round-robin)
items = list(range(12))
print(f"📦 Distributing {len(items)} items across workers...")

start = time.time()
futures = []
for i, item in enumerate(items):
    # Round-robin assignment
    worker = workers[i % NUM_WORKERS]
    futures.append(worker.process.remote(item))

results = ray.get(futures)
print(f"\n   Results:")
for r in results[:6]:
    print(f"   {r}")
print(f"   ...")

print(f"\n⏱️  Total time: {time.time() - start:.2f}s")
print(f"   (12 items, 4 workers, 0.3s each = 12/4 × 0.3 = 0.9s)")

# Check distribution
print(f"\n📊 Items processed per worker:")
for w in workers:
    count = ray.get(w.get_count.remote())
    print(f"   Worker {workers.index(w)}: {count} items")


# ============================================================================
# SECTION 5: PASSING ACTOR HANDLES
# ============================================================================
"""
Actor handles (references) can be passed to tasks or other actors!

This enables powerful patterns:
- Tasks that update shared state
- Actors that communicate with each other
- Hierarchical actor systems
"""

print("\n" + "=" * 60)
print("SECTION 5: Passing Actor Handles")
print("=" * 60)

@ray.remote
class ResultCollector:
    """Collects results from multiple workers."""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, result):
        self.results.append(result)
        return len(self.results)
    
    def get_all_results(self):
        return self.results

@ray.remote
def worker_task(task_id, collector):
    """Task that sends results to a collector actor."""
    time.sleep(random.uniform(0.1, 0.5))
    result = f"Result from task {task_id}"
    # Call method on the collector actor!
    count = ray.get(collector.add_result.remote(result))
    return f"Task {task_id} submitted (total: {count})"

print("\n📮 Creating result collector actor...")
collector = ResultCollector.remote()

print("🚀 Launching tasks that report to collector...")
futures = [worker_task.remote(i, collector) for i in range(5)]
results = ray.get(futures)

for r in results:
    print(f"   {r}")

print(f"\n📊 All collected results:")
all_results = ray.get(collector.get_all_results.remote())
for r in all_results:
    print(f"   - {r}")


# ============================================================================
# SECTION 6: NAMED ACTORS (Discoverable)
# ============================================================================
"""
Named actors can be looked up by name from anywhere in the cluster!

Use cases:
- Global services (rate limiter, config server)
- Singleton patterns
- Cross-driver communication

Options:
- name: Makes actor discoverable
- namespace: Isolates named actors (default: "default")
- lifetime: "detached" survives driver exit
"""

print("\n" + "=" * 60)
print("SECTION 6: Named Actors")
print("=" * 60)

@ray.remote
class GlobalConfig:
    """Configuration service - global singleton."""
    
    def __init__(self):
        self.config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "max_retries": 3
        }
    
    def get(self, key):
        return self.config.get(key)
    
    def set(self, key, value):
        self.config[key] = value
        return f"Set {key}={value}"
    
    def get_all(self):
        return self.config.copy()

# Create a named actor
print("\n🏷️  Creating named actor 'config_service'...")
try:
    # Try to get existing actor first
    config = ray.get_actor("config_service")
    print("   Found existing actor!")
except ValueError:
    # Doesn't exist, create it
    config = GlobalConfig.options(name="config_service").remote()
    print("   Created new actor!")

# Use it
print(f"\n📋 Current config: {ray.get(config.get_all.remote())}")

# Now, from "anywhere" in the cluster, we can look it up:
print("\n🔍 Looking up actor by name...")
same_config = ray.get_actor("config_service")
print(f"   Found it! batch_size = {ray.get(same_config.get.remote('batch_size'))}")

# Update config
ray.get(config.set.remote("batch_size", 64))
print(f"   Updated batch_size to 64")

# Verify from the looked-up handle
print(f"   Verified: batch_size = {ray.get(same_config.get.remote('batch_size'))}")


# ============================================================================
# SECTION 7: ASYNC ACTOR METHODS
# ============================================================================
"""
Actors can have async methods for I/O-bound operations.
Use async def for methods that do network calls, file I/O, etc.

Benefits:
- Better concurrency within a single actor
- Can handle multiple requests concurrently
- Ideal for I/O-bound workloads
"""

print("\n" + "=" * 60)
print("SECTION 7: Async Actor Methods")
print("=" * 60)

import asyncio

@ray.remote
class AsyncFetcher:
    """Actor with async methods for I/O operations."""
    
    def __init__(self):
        self.cache = {}
    
    async def fetch(self, url):
        """Simulated async fetch."""
        # Check cache first
        if url in self.cache:
            return {"url": url, "cached": True, "data": self.cache[url]}
        
        # Simulate async I/O
        await asyncio.sleep(0.5)
        data = f"Data from {url}"
        self.cache[url] = data
        return {"url": url, "cached": False, "data": data}

print("\n🌐 Creating async fetcher actor...")
fetcher = AsyncFetcher.remote()

# Fetch same URL twice
print("📡 Fetching URL (first time)...")
result1 = ray.get(fetcher.fetch.remote("https://api.example.com/data"))
print(f"   {result1}")

print("📡 Fetching URL (second time - should be cached)...")
result2 = ray.get(fetcher.fetch.remote("https://api.example.com/data"))
print(f"   {result2}")


# ============================================================================
# SECTION 8: PRACTICAL EXAMPLE - RATE LIMITER
# ============================================================================
"""
Real-world example: A distributed rate limiter using actors.

Requirements:
- Limit requests to N per second
- Track requests across all workers
- Thread-safe (actor guarantees this!)
"""

print("\n" + "=" * 60)
print("SECTION 8: Practical Example - Rate Limiter")
print("=" * 60)

@ray.remote
class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate_limit, time_window=1.0):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit
        self.last_update = time.time()
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * (self.rate_limit / self.time_window)
        self.tokens = min(self.rate_limit, self.tokens + new_tokens)
        self.last_update = now
    
    def acquire(self):
        """Try to acquire a token. Returns True if allowed."""
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
    
    def get_tokens(self):
        """Check available tokens."""
        self._refill()
        return self.tokens

@ray.remote
def make_api_call(call_id, rate_limiter):
    """Task that respects rate limiting."""
    if ray.get(rate_limiter.acquire.remote()):
        time.sleep(0.1)  # Simulate API call
        return f"Call {call_id}: ✅ Success"
    else:
        return f"Call {call_id}: ❌ Rate limited"

# Create rate limiter: 5 requests per second
print("\n⏱️  Creating rate limiter (5 requests/second)...")
limiter = RateLimiter.remote(rate_limit=5)

# Try to make 10 rapid requests
print("🚀 Making 10 rapid requests...")
futures = [make_api_call.remote(i, limiter) for i in range(10)]
results = ray.get(futures)

for r in results:
    print(f"   {r}")

print("\n⏳ Waiting 1 second for tokens to refill...")
time.sleep(1)

print(f"   Available tokens: {ray.get(limiter.get_tokens.remote()):.1f}")


# ============================================================================
# EXERCISES
# ============================================================================
"""
🏋️ EXERCISES - Master actors!

EXERCISE 1: Key-Value Store
    Create an actor that implements a key-value store:
    - put(key, value)
    - get(key) -> value or None
    - delete(key) -> bool
    - keys() -> list of keys

EXERCISE 2: Actor Pool with Load Balancing
    Implement an actor pool that:
    - Creates N worker actors
    - Tracks how many tasks each worker is processing
    - Assigns new tasks to the least busy worker
    Hint: Use a coordinator actor to track load.

EXERCISE 3: Pub/Sub System
    Create a simple publish-subscribe system:
    - Publisher actor: publish(topic, message)
    - Subscriber actors: subscribe(topic), receive messages
    - Topics: Multiple subscribers per topic

EXERCISE 4: Cache with TTL
    Extend the caching example:
    - Each cached item has a TTL (time-to-live)
    - Expired items return None
    - Add a cleanup() method to remove expired items

EXERCISE 5: Actor Supervision
    Create a "supervisor" actor that:
    - Manages a pool of worker actors
    - Monitors their health (ping every N seconds)
    - Restarts workers that stop responding
"""

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
🏋️ Complete the exercises above to master actors!

📝 Key takeaways:
   1. Actors = Stateful classes with @ray.remote
   2. Methods execute SEQUENTIALLY (no race conditions)
   3. Create multiple actors for parallelism (actor pools)
   4. Actor handles can be passed to tasks and other actors
   5. Named actors are discoverable from anywhere in the cluster
   6. Async methods enable I/O concurrency within an actor

🔜 Next lesson: 04_object_store.py
   - ray.put() and the distributed object store
   - Zero-copy and memory efficiency
   - Object passing patterns
""")

# Cleanup named actor (optional)
# ray.kill(config, no_restart=True)
