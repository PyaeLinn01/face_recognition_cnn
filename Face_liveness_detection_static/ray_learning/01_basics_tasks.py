"""
==============================================================================
LESSON 1: RAY BASICS & TASKS
==============================================================================

🎯 Learning Objectives:
    1. Initialize Ray and understand the cluster
    2. Convert regular functions to distributed tasks with @ray.remote
    3. Execute tasks asynchronously with .remote()
    4. Retrieve results with ray.get()
    5. Understand the difference between serial and parallel execution

⏱️ Estimated time: 20-30 minutes

Run this file: python 01_basics_tasks.py
==============================================================================
"""

import ray
import time
import os

# ============================================================================
# SECTION 1: INITIALIZING RAY
# ============================================================================
"""
ray.init() starts the Ray runtime:
- Creates a local cluster (head node + worker processes)
- Starts the distributed object store
- Launches the Ray Dashboard at http://127.0.0.1:8265

Key parameters:
- num_cpus: Override detected CPUs (useful for testing)
- num_gpus: Override detected GPUs
- logging_level: Control log verbosity
- dashboard_host: Dashboard binding address

For production clusters, you'd connect to an existing cluster:
    ray.init(address="ray://cluster-head:10001")
"""

print("=" * 60)
print("SECTION 1: Initializing Ray")
print("=" * 60)

# Initialize Ray - this starts the local cluster
# ignore_reinit_error=True allows re-running this script
ray.init(ignore_reinit_error=True)

# Let's see what resources Ray detected
print(f"\n📊 Cluster Resources:")
print(f"   CPUs available: {ray.cluster_resources().get('CPU', 0)}")
print(f"   GPUs available: {ray.cluster_resources().get('GPU', 0)}")
print(f"   Memory: {ray.cluster_resources().get('memory', 0) / 1e9:.2f} GB")
print(f"\n🌐 Dashboard: http://127.0.0.1:8265")


# ============================================================================
# SECTION 2: YOUR FIRST RAY TASK
# ============================================================================
"""
The @ray.remote decorator transforms a regular Python function into a 
"remote function" that can run on any worker in the cluster.

Key concepts:
1. @ray.remote - Decorator that registers the function with Ray
2. .remote() - Submits the function for execution (non-blocking!)
3. ray.get() - Blocks and retrieves the result

IMPORTANT: .remote() returns immediately with an ObjectRef (future/promise),
NOT the actual result! This is the key to parallelism.
"""

print("\n" + "=" * 60)
print("SECTION 2: Your First Ray Task")
print("=" * 60)

# Regular Python function (runs in current process)
def slow_square_regular(x):
    """Simulates a slow computation."""
    time.sleep(1)  # Pretend this takes 1 second
    return x * x

# Ray remote function (can run on any worker)
@ray.remote
def slow_square(x):
    """Same function, but distributed!"""
    time.sleep(1)  # Pretend this takes 1 second
    return x * x

# Let's compare execution

# --- Regular execution (sequential) ---
print("\n🐢 Regular Python (sequential):")
start = time.time()
results_regular = [slow_square_regular(i) for i in range(4)]
print(f"   Results: {results_regular}")
print(f"   Time: {time.time() - start:.2f}s (4 tasks × 1s = ~4s)")

# --- Ray execution (parallel) ---
print("\n🚀 Ray (parallel):")
start = time.time()

# Step 1: Submit all tasks (this returns IMMEDIATELY!)
futures = [slow_square.remote(i) for i in range(4)]
print(f"   Futures (ObjectRefs): {futures}")
print(f"   ⚡ Submission took: {time.time() - start:.4f}s (nearly instant!)")

# Step 2: Retrieve results (this BLOCKS until all complete)
results_ray = ray.get(futures)
print(f"   Results: {results_ray}")
print(f"   Total time: {time.time() - start:.2f}s (parallel = ~1s)")


# ============================================================================
# SECTION 3: UNDERSTANDING OBJECT REFERENCES (FUTURES)
# ============================================================================
"""
When you call function.remote(), Ray returns an ObjectRef (object reference).

An ObjectRef is like a "promise" or "future" - it's a pointer to a value
that will exist in the distributed object store once the task completes.

Key insight: ObjectRefs are LIGHTWEIGHT. You can pass them around, store them,
and even pass them to other tasks BEFORE the value exists!
"""

print("\n" + "=" * 60)
print("SECTION 3: Understanding ObjectRefs (Futures)")
print("=" * 60)

@ray.remote
def get_pid():
    """Returns the process ID where this task runs."""
    return os.getpid()

# Submit a task
future = get_pid.remote()

print(f"\n📦 ObjectRef: {future}")
print(f"   Type: {type(future)}")
print(f"   This is NOT the result - it's a reference to where the result will be!")

# Now get the actual value
pid = ray.get(future)
print(f"\n   Actual value (worker PID): {pid}")
print(f"   Main process PID: {os.getpid()}")
print(f"   (Notice they're different - the task ran in a separate worker!)")


# ============================================================================
# SECTION 4: TASK EXECUTION MODEL
# ============================================================================
"""
Understanding WHERE and HOW tasks execute:

1. Each task runs in a SEPARATE WORKER PROCESS
2. Workers are created by Ray and managed automatically
3. Multiple tasks can run in parallel (up to available CPUs)
4. Each task gets its own Python interpreter (no GIL issues!)

This is why Ray can achieve true parallelism - unlike Python threads,
each worker is a separate process with its own GIL.
"""

print("\n" + "=" * 60)
print("SECTION 4: Task Execution Model")
print("=" * 60)

@ray.remote
def identify_worker():
    """Identify which worker process handles this task."""
    import time
    time.sleep(0.5)  # Small delay to see parallelism
    return f"Worker PID: {os.getpid()}"

print("\n🔍 Launching 8 tasks to see worker distribution:")
futures = [identify_worker.remote() for _ in range(8)]
results = ray.get(futures)

for i, result in enumerate(results):
    print(f"   Task {i}: {result}")

# Count unique workers
unique_workers = len(set(results))
print(f"\n   📊 Tasks ran across {unique_workers} worker processes")


# ============================================================================
# SECTION 5: PASSING ARGUMENTS & RETURN VALUES
# ============================================================================
"""
Tasks can accept and return complex Python objects:
- Primitive types (int, float, str, bool)
- Collections (list, dict, tuple, set)
- NumPy arrays (with zero-copy optimization!)
- Custom classes (must be serializable)

Ray uses Apache Arrow for efficient serialization.
Large objects are stored in shared memory for zero-copy access.
"""

print("\n" + "=" * 60)
print("SECTION 5: Passing Arguments & Return Values")
print("=" * 60)

@ray.remote
def process_data(data_dict):
    """Process a dictionary and return transformed data."""
    return {
        "original_keys": list(data_dict.keys()),
        "sum_of_values": sum(data_dict.values()),
        "processed_by": os.getpid()
    }

input_data = {"a": 10, "b": 20, "c": 30}
print(f"\n📥 Input: {input_data}")

result = ray.get(process_data.remote(input_data))
print(f"📤 Output: {result}")


# ============================================================================
# SECTION 6: MULTIPLE RETURN VALUES
# ============================================================================
"""
Tasks can return multiple values using ray.remote(num_returns=N).
This creates N separate ObjectRefs, useful for:
- Splitting large results
- Pipelining (downstream tasks can start before all outputs ready)
"""

print("\n" + "=" * 60)
print("SECTION 6: Multiple Return Values")
print("=" * 60)

@ray.remote(num_returns=3)
def split_data(numbers):
    """Split numbers into three groups: negative, zero, positive."""
    negative = [x for x in numbers if x < 0]
    zero = [x for x in numbers if x == 0]
    positive = [x for x in numbers if x > 0]
    return negative, zero, positive

numbers = [-3, -1, 0, 0, 1, 5, 10, -7]
print(f"\n📊 Input numbers: {numbers}")

# Returns 3 ObjectRefs!
neg_ref, zero_ref, pos_ref = split_data.remote(numbers)

print(f"\n   Three separate ObjectRefs returned:")
print(f"   Negative ref: {neg_ref}")
print(f"   Zero ref: {zero_ref}")
print(f"   Positive ref: {pos_ref}")

# Get each result
print(f"\n   Negative: {ray.get(neg_ref)}")
print(f"   Zero: {ray.get(zero_ref)}")
print(f"   Positive: {ray.get(pos_ref)}")


# ============================================================================
# SECTION 7: TIMING COMPARISON - THE POWER OF PARALLELISM
# ============================================================================
"""
Let's do a proper benchmark to really see the speedup.
We'll simulate a CPU-intensive task and compare:
1. Regular Python (sequential)
2. Ray (parallel)
"""

print("\n" + "=" * 60)
print("SECTION 7: Timing Comparison - The Power of Parallelism")
print("=" * 60)

def cpu_intensive_regular(n):
    """Simulate CPU-intensive work."""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

@ray.remote
def cpu_intensive(n):
    """Same work, but can run in parallel."""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

NUM_TASKS = 8
WORK_SIZE = 1_000_000

print(f"\n⚙️  Running {NUM_TASKS} tasks, each processing {WORK_SIZE:,} iterations")

# Sequential
print("\n🐢 Sequential Python:")
start = time.time()
results_seq = [cpu_intensive_regular(WORK_SIZE) for _ in range(NUM_TASKS)]
seq_time = time.time() - start
print(f"   Time: {seq_time:.2f}s")

# Parallel with Ray
print("\n🚀 Parallel Ray:")
start = time.time()
futures = [cpu_intensive.remote(WORK_SIZE) for _ in range(NUM_TASKS)]
results_par = ray.get(futures)
par_time = time.time() - start
print(f"   Time: {par_time:.2f}s")

print(f"\n📈 Speedup: {seq_time / par_time:.2f}x faster!")
print(f"   (Theoretical max with {ray.cluster_resources().get('CPU', 1):.0f} CPUs: "
      f"{ray.cluster_resources().get('CPU', 1):.0f}x)")


# ============================================================================
# EXERCISES
# ============================================================================
"""
🏋️ EXERCISES - Try these to reinforce your learning!

EXERCISE 1: Parallel File Processor
    Create a Ray task that reads a text file and counts words.
    Process multiple files in parallel.
    
EXERCISE 2: Prime Number Finder
    Write a task that checks if a number is prime.
    Find all primes between 1 and 10000 in parallel.
    Compare with sequential execution.

EXERCISE 3: Nested Tasks
    Create a task that itself launches other tasks.
    Hint: You can call .remote() inside a @ray.remote function!
    
EXERCISE 4: Error Handling
    What happens when a task raises an exception?
    Try: create a task that divides by zero and call ray.get() on it.
    
EXERCISE 5: Dashboard Exploration
    Open http://127.0.0.1:8265 while running this script.
    Explore: Jobs, Actors, Metrics, Logs tabs.
"""

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
🏋️ Now it's your turn! Try the exercises above.

📝 Key takeaways from this lesson:
   1. ray.init() starts the local Ray cluster
   2. @ray.remote converts functions to distributed tasks
   3. .remote() submits tasks asynchronously (returns ObjectRef)
   4. ray.get() blocks and retrieves actual values
   5. Parallelism comes from running tasks on multiple workers

🔜 Next lesson: 02_parallel_patterns.py
   - Learn map-reduce patterns
   - Task dependencies and chaining
   - ray.wait() for advanced control
""")

# Cleanup (optional - Ray will cleanup on script exit)
# ray.shutdown()
