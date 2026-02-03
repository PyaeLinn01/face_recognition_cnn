"""
==============================================================================
LESSON 4: THE DISTRIBUTED OBJECT STORE
==============================================================================

🎯 Learning Objectives:
    1. Understand Ray's distributed object store architecture
    2. Master ray.put() for efficient data sharing
    3. Learn about zero-copy reads and shared memory
    4. Avoid common anti-patterns (serialization overhead)
    5. Object ownership and lifetime management
    6. Memory management and object spilling

⏱️ Estimated time: 35-45 minutes

Prerequisites: Complete lessons 01-03 first

Run this file: python 04_object_store.py
==============================================================================
"""

import ray
import time
import numpy as np
import sys

ray.init(ignore_reinit_error=True)

# ============================================================================
# SECTION 1: THE OBJECT STORE ARCHITECTURE
# ============================================================================
"""
Ray's Object Store is the MAGIC that makes distributed computing feel local.

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Ray Cluster                                         │
├───────────────────────────────────────────────────────────────────────────── │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐     │
│   │     Node 1       │    │     Node 2       │    │     Node 3       │     │
│   ├──────────────────┤    ├──────────────────┤    ├──────────────────┤     │
│   │  Object Store    │←──→│  Object Store    │←──→│  Object Store    │     │
│   │  (Shared Mem)    │    │  (Shared Mem)    │    │  (Shared Mem)    │     │
│   ├──────────────────┤    ├──────────────────┤    ├──────────────────┤     │
│   │  Worker  Worker  │    │  Worker  Worker  │    │  Worker  Worker  │     │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘

Key properties:
1. IMMUTABLE objects (write once, read many)
2. SHARED MEMORY access on same node (zero-copy for workers)
3. AUTOMATIC FETCH from remote nodes when needed
4. REFERENCE COUNTING for garbage collection
5. SPILLING to disk when memory is full
"""

print("=" * 60)
print("SECTION 1: Object Store Overview")
print("=" * 60)

# Check object store memory
available = ray.cluster_resources()
print(f"\n📊 Object Store Status:")
print(f"   Available memory: {available.get('object_store_memory', 0) / 1e9:.2f} GB")


# ============================================================================
# SECTION 2: ray.put() - EXPLICIT OBJECT CREATION
# ============================================================================
"""
ray.put(value) -> ObjectRef

Places a value in the object store and returns an ObjectRef.

When to use ray.put():
1. Share LARGE data across multiple tasks
2. Avoid repeated serialization
3. Control when data is stored in the object store

Without ray.put(): Data is serialized EACH TIME you pass it to a task.
With ray.put(): Data is stored ONCE and only the reference is passed.
"""

print("\n" + "=" * 60)
print("SECTION 2: ray.put() - Efficient Data Sharing")
print("=" * 60)

@ray.remote
def process_array(arr):
    """Sum all elements of an array."""
    return np.sum(arr)

# Create a large array
large_array = np.ones((10000, 10000))  # ~800 MB
print(f"\n📦 Large array size: {large_array.nbytes / 1e6:.0f} MB")

# ❌ BAD: Pass data directly (serialized 5 times!)
print("\n❌ BAD PATTERN: Passing data directly...")
start = time.time()
futures = [process_array.remote(large_array) for _ in range(5)]
results = ray.get(futures)
bad_time = time.time() - start
print(f"   Time: {bad_time:.2f}s (array serialized 5 times!)")

# ✅ GOOD: Use ray.put() (serialized once!)
print("\n✅ GOOD PATTERN: Using ray.put()...")
start = time.time()
array_ref = ray.put(large_array)  # Store once in object store
futures = [process_array.remote(array_ref) for _ in range(5)]
results = ray.get(futures)
good_time = time.time() - start
print(f"   Time: {good_time:.2f}s (array stored once, reference passed)")

print(f"\n📈 Speedup: {bad_time / good_time:.1f}x faster!")


# ============================================================================
# SECTION 3: OBJECT REFERENCES (ObjectRef)
# ============================================================================
"""
ObjectRef is Ray's pointer to data in the object store.

Key properties:
- Lightweight (just an ID, ~20 bytes)
- Can be passed between tasks/actors
- Can be used before the object exists (futures)
- Automatically fetched when needed
"""

print("\n" + "=" * 60)
print("SECTION 3: Understanding ObjectRef")
print("=" * 60)

# Create an object and get its reference
data = {"key": "value", "numbers": [1, 2, 3]}
ref = ray.put(data)

print(f"\n📌 ObjectRef: {ref}")
print(f"   Type: {type(ref)}")
print(f"   Hex ID: {ref.hex()}")

# Get the object back
retrieved = ray.get(ref)
print(f"\n📤 Retrieved data: {retrieved}")
print(f"   Same as original? {retrieved == data}")


# ============================================================================
# SECTION 4: ZERO-COPY READS (SHARED MEMORY)
# ============================================================================
"""
For NumPy arrays, Ray uses Apache Arrow for ZERO-COPY reads.

This means:
- Array is stored in shared memory
- Workers on same node access it directly (no copy!)
- Massive performance improvement for large arrays

Zero-copy works for:
- NumPy arrays (ndarray)
- Apache Arrow tables
- Pandas DataFrames (converted to Arrow)
"""

print("\n" + "=" * 60)
print("SECTION 4: Zero-Copy Reads")
print("=" * 60)

@ray.remote
def check_memory_address(arr_ref):
    """Check the memory address of the array."""
    arr = arr_ref  # Ray resolves the reference
    return {
        "shape": arr.shape,
        "data_ptr": arr.ctypes.data,  # Memory address
        "pid": os.getpid()
    }

import os

# Create array and store in object store
print("\n🔬 Testing zero-copy behavior...")
test_array = np.random.rand(1000, 1000)
arr_ref = ray.put(test_array)

# Call from multiple workers
results = ray.get([check_memory_address.remote(arr_ref) for _ in range(3)])

print(f"\n   Workers accessing the array:")
for i, r in enumerate(results):
    print(f"   Worker {i} (PID {r['pid']}): data_ptr = {r['data_ptr']}")

print("\n   Note: On same node, workers share the same memory!")
print("   (Different PIDs, but potentially same data pointer = zero-copy)")


# ============================================================================
# SECTION 5: IMPLICIT vs EXPLICIT OBJECT CREATION
# ============================================================================
"""
Objects enter the object store in two ways:

IMPLICIT (automatic):
- Return values from tasks → automatically stored
- Arguments to tasks → serialized and stored

EXPLICIT (manual with ray.put()):
- You control when data is stored
- Best for data used by multiple tasks

Rule of thumb:
- Use ray.put() when data is used by 2+ tasks
- Let Ray handle it for single-use data
"""

print("\n" + "=" * 60)
print("SECTION 5: Implicit vs Explicit Object Creation")
print("=" * 60)

@ray.remote
def generate_data():
    """Generate data - return value is implicitly stored."""
    return np.random.rand(100, 100)

@ray.remote
def consume_data(data):
    """Consume data."""
    return np.sum(data)

# Implicit: return value stored automatically
print("\n📥 Implicit object creation (return values):")
data_ref = generate_data.remote()  # Returns ObjectRef
print(f"   generate_data() returned: {data_ref}")
print(f"   (Result is in object store, we have a reference)")

# We can pass this reference to another task
result = ray.get(consume_data.remote(data_ref))
print(f"   consume_data() result: {result:.2f}")


# ============================================================================
# SECTION 6: PASSING OBJECT REFERENCES
# ============================================================================
"""
ObjectRefs can be passed to tasks before the object exists!

Ray will automatically wait for the object to be available.
This enables elegant pipeline patterns.
"""

print("\n" + "=" * 60)
print("SECTION 6: Passing ObjectRefs as Arguments")
print("=" * 60)

@ray.remote
def stage1():
    """First stage: generate data."""
    print("   Stage 1: Generating data...")
    time.sleep(1)
    return {"data": [1, 2, 3, 4, 5]}

@ray.remote
def stage2(data_ref):
    """Second stage: process data (waits for stage1)."""
    # Ray automatically resolves data_ref!
    data = data_ref
    print(f"   Stage 2: Processing {data}")
    time.sleep(0.5)
    return {"processed": [x * 2 for x in data["data"]]}

@ray.remote
def stage3(data_ref):
    """Third stage: finalize (waits for stage2)."""
    data = data_ref
    print(f"   Stage 3: Finalizing {data}")
    return sum(data["processed"])

print("\n🔗 Building pipeline with ObjectRef passing:")

# Build the pipeline (no blocking!)
ref1 = stage1.remote()
ref2 = stage2.remote(ref1)  # Pass ref1 before it exists!
ref3 = stage3.remote(ref2)  # Pass ref2 before it exists!

print("   Pipeline built! Now executing...")
result = ray.get(ref3)
print(f"   Final result: {result}")


# ============================================================================
# SECTION 7: NESTED OBJECT REFERENCES
# ============================================================================
"""
Be careful with nested ObjectRefs!

If a task returns an ObjectRef, you need two ray.get() calls:
1. First ray.get() → gets the ObjectRef
2. Second ray.get() → gets the actual data

To avoid this, use ray.get() inside the task, or use
nested_ref = ray.get(outer_ref) followed by ray.get(nested_ref).
"""

print("\n" + "=" * 60)
print("SECTION 7: Nested Object References")
print("=" * 60)

@ray.remote
def create_inner():
    """Create data and return it."""
    return "inner data"

@ray.remote
def create_outer():
    """Create an ObjectRef and return it."""
    inner_ref = create_inner.remote()
    return inner_ref  # Returns ObjectRef, not data!

print("\n⚠️  Nested ObjectRef example:")
outer_ref = create_outer.remote()
print(f"   outer_ref: {outer_ref}")

# First ray.get() gives us another ObjectRef!
inner_ref = ray.get(outer_ref)
print(f"   ray.get(outer_ref): {inner_ref} (another ObjectRef!)")

# Second ray.get() gives us the actual data
data = ray.get(inner_ref)
print(f"   ray.get(inner_ref): {data} (the actual data!)")

# Better approach: resolve inside the task
@ray.remote
def create_outer_resolved():
    """Create and resolve the inner task."""
    inner_ref = create_inner.remote()
    return ray.get(inner_ref)  # Resolve inside!

print("\n✅ Better approach (resolve inside):")
outer_ref = create_outer_resolved.remote()
data = ray.get(outer_ref)
print(f"   Single ray.get(): {data}")


# ============================================================================
# SECTION 8: OBJECT OWNERSHIP AND LIFETIME
# ============================================================================
"""
Every object in Ray has an OWNER:
- Task return values → owned by the CALLER
- ray.put() objects → owned by the CALLER

Objects are garbage collected when:
1. All ObjectRefs go out of scope
2. Owner process (driver/task) exits

Important implications:
- Long-running tasks: objects stay alive
- Driver exit: all objects are cleaned up
"""

print("\n" + "=" * 60)
print("SECTION 8: Object Ownership")
print("=" * 60)

@ray.remote
def create_object():
    """Create an object - owned by caller."""
    return np.ones((100, 100))

print("\n📋 Object ownership:")
print("   - Object from ray.put() → owned by caller")
print("   - Object from task.remote() → owned by caller")
print("   - When owner exits, objects are garbage collected")

# Objects stay alive as long as we hold references
ref1 = ray.put("data 1")
ref2 = create_object.remote()

print(f"\n   ref1 (ray.put): {ref1}")
print(f"   ref2 (task return): {ref2}")
print("   Both owned by this driver process")


# ============================================================================
# SECTION 9: MEMORY MANAGEMENT AND SPILLING
# ============================================================================
"""
When the object store fills up, Ray SPILLS objects to disk.

Spilling order:
1. Objects are spilled to local disk
2. Can configure external storage (S3, GCS)
3. Objects are fetched back when needed

Memory pressure handling:
- LRU eviction for objects that can be recomputed
- Spilling for objects that must be preserved
"""

print("\n" + "=" * 60)
print("SECTION 9: Memory Management")
print("=" * 60)

print("\n💾 Object store memory management:")
print("   1. Objects stored in shared memory (fast access)")
print("   2. When memory fills up, Ray spills to disk")
print("   3. Spilled objects are fetched back when needed")
print("   4. LRU eviction for recreatable objects")

# You can configure spilling:
print("\n⚙️  Configuration options:")
print('   ray.init(object_store_memory=10*1024*1024*1024)  # 10GB')
print('   ray.init(_system_config={"object_spilling_threshold": 0.8})')


# ============================================================================
# SECTION 10: ANTI-PATTERNS TO AVOID
# ============================================================================
"""
Common mistakes with the object store:
"""

print("\n" + "=" * 60)
print("SECTION 10: Anti-Patterns to Avoid")
print("=" * 60)

print("""
❌ ANTI-PATTERN 1: Repeated serialization
   # Bad: array serialized 100 times
   futures = [task.remote(large_array) for _ in range(100)]
   
   # Good: array stored once
   ref = ray.put(large_array)
   futures = [task.remote(ref) for _ in range(100)]

❌ ANTI-PATTERN 2: Fetching objects you don't need
   # Bad: fetch just to pass to another task
   data = ray.get(task1.remote())
   result = ray.get(task2.remote(data))
   
   # Good: pass the reference directly
   ref = task1.remote()
   result = ray.get(task2.remote(ref))

❌ ANTI-PATTERN 3: Creating many small objects
   # Bad: 10000 tiny objects
   refs = [ray.put(i) for i in range(10000)]
   
   # Good: batch into larger objects
   ref = ray.put(list(range(10000)))

❌ ANTI-PATTERN 4: Holding references too long
   # Objects stay in memory as long as you hold refs
   # Let refs go out of scope when done
   
❌ ANTI-PATTERN 5: Putting non-serializable objects
   # Ray uses pickle/cloudpickle for serialization
   # Lambda functions, file handles, locks don't serialize well
""")


# ============================================================================
# SECTION 11: PRACTICAL EXAMPLE - DATA BROADCAST
# ============================================================================
"""
A common pattern: broadcast large data to all workers.

Example: Distribute a model/config to all inference tasks.
"""

print("\n" + "=" * 60)
print("SECTION 11: Practical Example - Data Broadcast")
print("=" * 60)

@ray.remote
def inference_task(model_ref, batch):
    """Run inference using a shared model."""
    model = model_ref  # Model is fetched once per worker!
    # Simulate inference
    return {"batch": batch, "predictions": [x * model["weight"] for x in batch]}

# Large model that we want to share
large_model = {"weight": 2.5, "params": np.random.rand(1000, 1000)}
print(f"\n🤖 Model size: {sys.getsizeof(large_model) / 1e6:.1f}+ MB")

# Broadcast model to object store
print("📡 Broadcasting model to object store...")
model_ref = ray.put(large_model)

# Run inference on multiple batches
batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
print(f"📊 Running inference on {len(batches)} batches...")

futures = [inference_task.remote(model_ref, batch) for batch in batches]
results = ray.get(futures)

for r in results:
    print(f"   Batch {r['batch']} → Predictions {r['predictions']}")

print("\n   ✅ Model was stored once, accessed by all workers!")


# ============================================================================
# EXERCISES
# ============================================================================
"""
🏋️ EXERCISES - Master the object store!

EXERCISE 1: Measure Serialization Overhead
    Create a 100MB numpy array.
    Compare time to run 10 tasks:
    a) Passing array directly
    b) Using ray.put()
    Calculate the serialization overhead.

EXERCISE 2: Object Store Memory Tracker
    Write a function that:
    - Creates objects of various sizes
    - Reports object store memory usage after each
    - Hint: Use ray.cluster_resources()

EXERCISE 3: Pipeline with Intermediate Objects
    Create a 3-stage pipeline where each stage:
    - Receives data from previous stage (ObjectRef)
    - Transforms the data
    - Returns new ObjectRef
    Measure: total time, intermediate object sizes

EXERCISE 4: Broadcast vs Point-to-Point
    Compare two patterns:
    a) One ray.put() + N tasks using the ref (broadcast)
    b) N tasks each creating their own copy
    Measure memory usage and time.

EXERCISE 5: Object Cleanup
    Demonstrate object lifecycle:
    - Create object with ray.put()
    - Pass to a task that stores the ref
    - Delete your local ref
    - Does the object get garbage collected? Why/why not?
"""

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
🏋️ Complete the exercises to master the object store!

📝 Key takeaways:
   1. ray.put() stores data ONCE in the object store
   2. ObjectRefs are lightweight pointers (~20 bytes)
   3. Zero-copy reads for NumPy arrays (shared memory)
   4. Pass ObjectRefs instead of data to avoid serialization
   5. Objects have owners and are garbage collected
   6. Memory spills to disk when object store is full

🔜 Next lesson: 05_resources_scheduling.py
   - CPU/GPU resource allocation
   - Custom resources
   - Placement groups and scheduling strategies
""")
