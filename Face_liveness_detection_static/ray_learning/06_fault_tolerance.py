"""
==============================================================================
LESSON 6: FAULT TOLERANCE
==============================================================================

🎯 Learning Objectives:
    1. Understand task retry mechanisms
    2. Configure actor restart policies
    3. Handle exceptions properly in distributed systems
    4. Implement checkpointing patterns
    5. Deal with node failures gracefully
    6. Design resilient distributed applications

⏱️ Estimated time: 40-50 minutes

Prerequisites: Complete lessons 01-05 first

Run this file: python 06_fault_tolerance.py
==============================================================================
"""

import ray
import time
import random
import os

ray.init(ignore_reinit_error=True)

# ============================================================================
# SECTION 1: WHY FAULT TOLERANCE MATTERS
# ============================================================================
"""
In distributed systems, failures are NOT exceptional - they're expected!

Types of failures:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Task-level failures:                                                        │
│   - Exceptions in code                                                      │
│   - Out-of-memory errors                                                    │
│   - Timeouts                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Actor-level failures:                                                       │
│   - Actor process crashes                                                   │
│   - Unhandled exceptions                                                    │
│   - Memory leaks                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Node-level failures:                                                        │
│   - Hardware failures                                                       │
│   - Network partitions                                                      │
│   - Spot instance preemption                                                │
│   - Node eviction                                                           │
└─────────────────────────────────────────────────────────────────────────────┘

Ray provides mechanisms to handle all of these!
"""

print("=" * 60)
print("SECTION 1: Why Fault Tolerance Matters")
print("=" * 60)

print("""
🔥 In distributed systems, failures happen:
   - Tasks may fail randomly
   - Actors may crash
   - Nodes may disappear
   
   Ray provides tools to handle all of these gracefully!
""")


# ============================================================================
# SECTION 2: TASK RETRIES
# ============================================================================
"""
Tasks can be automatically retried on failure.

@ray.remote(max_retries=N)
- Default: max_retries=3
- max_retries=-1: Retry infinitely
- max_retries=0: No retries

By default, only SYSTEM errors trigger retries.
To retry on APPLICATION errors (exceptions):
    @ray.remote(max_retries=3, retry_exceptions=True)

Or specify which exceptions to retry:
    @ray.remote(max_retries=3, retry_exceptions=[ValueError, TimeoutError])
"""

print("\n" + "=" * 60)
print("SECTION 2: Task Retries")
print("=" * 60)

# Track attempts with a global counter (for demo purposes)
attempt_counter = {}

@ray.remote(max_retries=3, retry_exceptions=True)
def flaky_task(task_id):
    """Task that fails 2/3 of the time, then succeeds."""
    # Track attempts
    key = f"task_{task_id}"
    if key not in attempt_counter:
        attempt_counter[key] = 0
    attempt_counter[key] += 1
    
    attempt = attempt_counter[key]
    
    # Fail first 2 attempts
    if attempt < 3:
        print(f"   Task {task_id}, attempt {attempt}: 💥 Failing...")
        raise ValueError(f"Simulated failure on attempt {attempt}")
    
    print(f"   Task {task_id}, attempt {attempt}: ✅ Success!")
    return f"Task {task_id} completed on attempt {attempt}"

print("\n🔄 Running flaky task with max_retries=3:")
try:
    result = ray.get(flaky_task.remote(1))
    print(f"   Result: {result}")
except ray.exceptions.RayTaskError as e:
    print(f"   Failed after all retries: {e}")


# Retrying specific exceptions
print("\n🎯 Retrying specific exceptions:")

@ray.remote(max_retries=3, retry_exceptions=[ValueError])
def selective_retry_task():
    """Only retries on ValueError, not on TypeError."""
    if random.random() < 0.5:
        raise ValueError("This will be retried")
    else:
        raise TypeError("This will NOT be retried")

print("   @ray.remote(max_retries=3, retry_exceptions=[ValueError])")
print("   - ValueError → retried")
print("   - TypeError → fails immediately")


# ============================================================================
# SECTION 3: HANDLING TASK EXCEPTIONS
# ============================================================================
"""
When a task fails (after all retries), ray.get() raises:
    ray.exceptions.RayTaskError

This wraps the original exception for inspection.

Best practices:
1. Catch RayTaskError around ray.get()
2. Access original exception via .cause attribute
3. Use try/except INSIDE tasks for recoverable errors
"""

print("\n" + "=" * 60)
print("SECTION 3: Handling Task Exceptions")
print("=" * 60)

@ray.remote(max_retries=0)  # No retries for this demo
def failing_task():
    """Task that always fails."""
    raise RuntimeError("Something went wrong!")

@ray.remote
def task_with_internal_handling():
    """Task that handles errors internally."""
    try:
        # Risky operation
        result = 1 / 0
    except ZeroDivisionError:
        # Handle internally, return error result
        return {"status": "error", "message": "Division by zero"}
    return {"status": "success", "result": result}

print("\n⚠️ External exception handling:")
try:
    result = ray.get(failing_task.remote())
except ray.exceptions.RayTaskError as e:
    print(f"   Caught RayTaskError!")
    print(f"   Original exception type: {type(e.cause).__name__}")
    print(f"   Message: {e.cause}")

print("\n✅ Internal exception handling:")
result = ray.get(task_with_internal_handling.remote())
print(f"   Result: {result}")


# ============================================================================
# SECTION 4: ACTOR FAULT TOLERANCE
# ============================================================================
"""
Actors have different fault tolerance mechanisms:

1. max_restarts: How many times to restart crashed actor
2. max_task_retries: Retry pending calls after restart
3. max_pending_calls: Queue size for calls during restart

@ray.remote(max_restarts=5, max_task_retries=3)
class ResilientActor:
    ...

IMPORTANT: When an actor restarts:
- __init__ is called again (state is LOST!)
- Pending method calls can be retried
- You need external checkpointing for state recovery
"""

print("\n" + "=" * 60)
print("SECTION 4: Actor Fault Tolerance")
print("=" * 60)

@ray.remote(max_restarts=3)
class FragileActor:
    """Actor that crashes occasionally."""
    
    def __init__(self):
        self.call_count = 0
        self.instance_id = random.randint(1000, 9999)
        print(f"   [Actor] Initialized with instance_id={self.instance_id}")
    
    def process(self):
        self.call_count += 1
        
        # Crash on 3rd call
        if self.call_count == 3:
            print(f"   [Actor {self.instance_id}] 💥 Crashing on call {self.call_count}!")
            # Simulate crash by raising an error
            os._exit(1)  # Force kill the process
        
        return f"Call {self.call_count} on instance {self.instance_id}"

print("\n🎭 Creating actor with max_restarts=3:")
actor = FragileActor.remote()

print("   Making calls to actor...")
for i in range(5):
    try:
        result = ray.get(actor.process.remote(), timeout=5)
        print(f"   Call {i+1}: {result}")
    except Exception as e:
        print(f"   Call {i+1}: Error - {type(e).__name__}")
    time.sleep(0.5)

print("\n   Notice: After crash, actor restarted with NEW instance_id")
print("   State (call_count) was reset - you need checkpointing for state!")


# ============================================================================
# SECTION 5: CHECKPOINTING PATTERN
# ============================================================================
"""
For true resilience, implement CHECKPOINTING:
1. Periodically save state to external storage
2. Restore from checkpoint on restart

Storage options:
- Local disk (single-node only)
- Redis, PostgreSQL
- Cloud storage (S3, GCS)
- Distributed cache
"""

print("\n" + "=" * 60)
print("SECTION 5: Checkpointing Pattern")
print("=" * 60)

import json
import tempfile

# Simulated external storage (would be Redis/S3 in production)
CHECKPOINT_DIR = tempfile.mkdtemp()

@ray.remote(max_restarts=3)
class CheckpointedCounter:
    """Actor with external checkpointing."""
    
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{name}_checkpoint.json")
        self._restore_checkpoint()
    
    def _restore_checkpoint(self):
        """Restore state from checkpoint on startup."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
                self.count = state["count"]
                print(f"   [Actor {self.name}] Restored from checkpoint: count={self.count}")
        else:
            print(f"   [Actor {self.name}] No checkpoint found, starting fresh")
    
    def _save_checkpoint(self):
        """Save state to checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({"count": self.count}, f)
    
    def increment(self):
        self.count += 1
        # Checkpoint every 5 increments
        if self.count % 5 == 0:
            self._save_checkpoint()
            print(f"   [Actor {self.name}] Checkpointed at count={self.count}")
        return self.count
    
    def get_count(self):
        return self.count
    
    def force_checkpoint(self):
        """Manually trigger checkpoint."""
        self._save_checkpoint()
        return f"Checkpointed at count={self.count}"

print("\n📸 Creating checkpointed actor:")
actor = CheckpointedCounter.remote("my_counter")

print("   Incrementing 12 times...")
for i in range(12):
    count = ray.get(actor.increment.remote())
    if i == 11:
        print(f"   Final count: {count}")

# Force checkpoint before "crash"
ray.get(actor.force_checkpoint.remote())

print("\n   Simulating crash and restart...")
ray.kill(actor)
time.sleep(1)

# Create new actor with same name
actor2 = CheckpointedCounter.remote("my_counter")
count = ray.get(actor2.get_count.remote())
print(f"   After restart, count restored to: {count}")


# ============================================================================
# SECTION 6: OBJECT FAULT TOLERANCE
# ============================================================================
"""
What happens when objects are lost (node failure)?

Ray uses LINEAGE RECONSTRUCTION:
1. Objects store their "lineage" (how they were created)
2. If an object is lost, Ray re-executes the creating task
3. This is automatic and transparent

Limitations:
- Only works if the creating task is deterministic
- May cause cascading re-execution
- For critical data, use explicit checkpointing
"""

print("\n" + "=" * 60)
print("SECTION 6: Object Fault Tolerance")
print("=" * 60)

print("""
🔄 Lineage Reconstruction:

   Task A → Object X → Task B → Object Y
   
   If Object X is lost (node dies):
   1. Ray detects the loss
   2. Re-executes Task A
   3. Recreates Object X
   4. Task B can continue

   ⚠️ Caveats:
   - Task A must be deterministic (same input → same output)
   - If Task A has side effects, they may be duplicated
   - For large pipelines, use explicit checkpointing
""")


# ============================================================================
# SECTION 7: TIMEOUT HANDLING
# ============================================================================
"""
Timeouts prevent indefinite waiting.

1. ray.get(ref, timeout=N): Raise TimeoutError after N seconds
2. ray.wait(..., timeout=N): Return available results after N seconds
3. Task-level timeouts (via signal handling inside task)
"""

print("\n" + "=" * 60)
print("SECTION 7: Timeout Handling")
print("=" * 60)

@ray.remote
def slow_task():
    """Task that takes a long time."""
    time.sleep(10)
    return "Done after 10 seconds"

print("\n⏰ Testing timeout with ray.get():")
try:
    result = ray.get(slow_task.remote(), timeout=2)
    print(f"   Result: {result}")
except ray.exceptions.GetTimeoutError:
    print("   TimeoutError: Task didn't complete in 2 seconds")
    print("   (Task is still running in background!)")

print("\n⏰ Testing timeout with ray.wait():")
futures = [slow_task.remote() for _ in range(3)]
done, not_done = ray.wait(futures, num_returns=3, timeout=1)
print(f"   After 1 second: {len(done)} done, {len(not_done)} still running")


# ============================================================================
# SECTION 8: GRACEFUL DEGRADATION PATTERN
# ============================================================================
"""
Design for partial failures - don't let one failure cascade!

Patterns:
1. Circuit breaker: Stop calling failing services
2. Fallback: Return default/cached value on failure
3. Bulkhead: Isolate failures to prevent cascade
"""

print("\n" + "=" * 60)
print("SECTION 8: Graceful Degradation Pattern")
print("=" * 60)

@ray.remote
class CircuitBreaker:
    """Actor that implements circuit breaker pattern."""
    
    def __init__(self, failure_threshold=3, reset_timeout=10):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = "CLOSED"  # CLOSED (normal), OPEN (failing), HALF-OPEN (testing)
        self.last_failure_time = None
    
    def call(self, func, *args):
        """Call function with circuit breaker protection."""
        # Check if circuit is open
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF-OPEN"
            else:
                return {"status": "circuit_open", "fallback": "cached_result"}
        
        try:
            result = func(*args)
            # Success - reset failure count
            self.failure_count = 0
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
            return {"status": "success", "result": result}
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                return {"status": "circuit_opened", "error": str(e)}
            
            return {"status": "failed", "error": str(e)}
    
    def get_state(self):
        return {
            "state": self.state,
            "failure_count": self.failure_count
        }

def risky_function(should_fail):
    """Function that may fail."""
    if should_fail:
        raise RuntimeError("Service unavailable")
    return "Success!"

print("\n🔌 Circuit Breaker Pattern:")
cb = CircuitBreaker.remote(failure_threshold=3)

# Simulate failures
print("   Simulating 4 failures...")
for i in range(4):
    result = ray.get(cb.call.remote(risky_function, True))
    state = ray.get(cb.get_state.remote())
    print(f"   Call {i+1}: status={result['status']}, circuit={state['state']}")

# Now the circuit is open
print("\n   Circuit is OPEN - calls return fallback:")
result = ray.get(cb.call.remote(risky_function, False))
print(f"   Result: {result}")


# ============================================================================
# SECTION 9: PRODUCTION FAULT TOLERANCE CHECKLIST
# ============================================================================
"""
Checklist for production-ready fault tolerance:
"""

print("\n" + "=" * 60)
print("SECTION 9: Production Checklist")
print("=" * 60)

print("""
✅ PRODUCTION FAULT TOLERANCE CHECKLIST:

TASKS:
□ Set appropriate max_retries for each task type
□ Use retry_exceptions for transient failures
□ Implement idempotent tasks where possible
□ Add timeouts to prevent infinite waits

ACTORS:
□ Set max_restarts for critical actors
□ Implement checkpointing for stateful actors
□ Use external storage (Redis, S3) for checkpoints
□ Handle actor restart in calling code

OBJECTS:
□ Use ray.put() for frequently accessed large objects
□ Consider explicit checkpointing for critical data
□ Understand lineage reconstruction limitations

ERROR HANDLING:
□ Catch RayTaskError around ray.get()
□ Implement circuit breakers for external services
□ Log failures for debugging
□ Monitor failure rates

CLUSTER:
□ Use multiple nodes for redundancy
□ Configure autoscaler for capacity
□ Plan for node preemption (spot instances)
□ Test failover scenarios regularly
""")


# ============================================================================
# SECTION 10: PRACTICAL EXAMPLE - RESILIENT DATA PIPELINE
# ============================================================================
"""
Putting it all together: A fault-tolerant data processing pipeline.
"""

print("\n" + "=" * 60)
print("SECTION 10: Resilient Data Pipeline Example")
print("=" * 60)

@ray.remote(max_retries=3, retry_exceptions=True)
def fetch_data(source_id):
    """Fetch data from a source (may fail)."""
    if random.random() < 0.3:  # 30% failure rate
        raise ConnectionError(f"Failed to connect to source {source_id}")
    time.sleep(0.2)
    return {"source": source_id, "data": list(range(10))}

@ray.remote(max_retries=2, retry_exceptions=True)
def process_data(data):
    """Process data (may fail)."""
    if random.random() < 0.2:  # 20% failure rate
        raise ValueError("Processing error")
    time.sleep(0.1)
    return {"processed": sum(data["data"]), "source": data["source"]}

@ray.remote
def save_results(results, failed_sources):
    """Save results and report on failures."""
    return {
        "successful": len(results),
        "failed": len(failed_sources),
        "failed_sources": failed_sources
    }

print("\n🚀 Running resilient pipeline on 10 sources...")

# Fetch phase with error collection
fetch_futures = {i: fetch_data.remote(i) for i in range(10)}
fetch_results = []
fetch_failures = []

for source_id, future in fetch_futures.items():
    try:
        result = ray.get(future, timeout=5)
        fetch_results.append(result)
    except Exception as e:
        fetch_failures.append(source_id)
        print(f"   Source {source_id}: ❌ Failed after retries")

print(f"   Fetch phase: {len(fetch_results)} success, {len(fetch_failures)} failed")

# Process phase
process_futures = [process_data.remote(data) for data in fetch_results]
process_results = []
process_failures = []

for i, future in enumerate(process_futures):
    try:
        result = ray.get(future, timeout=5)
        process_results.append(result)
    except Exception as e:
        process_failures.append(fetch_results[i]["source"])

print(f"   Process phase: {len(process_results)} success, {len(process_failures)} failed")

# Final report
total_failed = list(set(fetch_failures + process_failures))
print(f"\n📊 Pipeline Summary:")
print(f"   Total sources: 10")
print(f"   Successfully processed: {len(process_results)}")
print(f"   Failed sources: {total_failed}")


# ============================================================================
# EXERCISES
# ============================================================================
"""
🏋️ EXERCISES - Master fault tolerance!

EXERCISE 1: Retry with Exponential Backoff
    Implement a task that retries with exponential backoff:
    - Wait 1s, 2s, 4s, 8s between retries
    - Give up after 4 retries
    - Hint: Track attempts with ray.put() or an actor

EXERCISE 2: Stateful Actor with Redis
    Create an actor that:
    - Stores state in Redis (or simulated)
    - Checkpoints every N operations
    - Restores from Redis on restart
    - Test by killing and recreating the actor

EXERCISE 3: Partial Result Collection
    Launch 20 tasks with varying success rates.
    Use ray.wait() to:
    - Collect results as they complete
    - Stop after 15 successful results
    - Report which tasks failed

EXERCISE 4: Actor Supervisor
    Implement a supervisor actor that:
    - Creates and monitors N worker actors
    - Detects when workers crash
    - Restarts crashed workers
    - Redistributes failed work

EXERCISE 5: Idempotent Task Design
    Design an idempotent data processing task:
    - Same input always produces same output
    - Safe to retry without side effects
    - Uses external deduplication (check if already processed)
"""

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
🏋️ Complete the exercises to master fault tolerance!

📝 Key takeaways:
   1. max_retries controls automatic task retries
   2. retry_exceptions enables retry on application errors
   3. Actors need max_restarts + checkpointing for resilience
   4. Objects use lineage reconstruction (automatic)
   5. Use timeouts to prevent infinite waits
   6. Design for partial failures (circuit breakers, fallbacks)

🎉 CONGRATULATIONS! You've completed Ray Core fundamentals!

📚 Next steps:
   - Build a real project using these concepts
   - Explore Ray libraries: Ray Data, Train, Tune, Serve
   - Deploy a Ray cluster on Kubernetes
   - Read: docs.ray.io for advanced topics
""")
