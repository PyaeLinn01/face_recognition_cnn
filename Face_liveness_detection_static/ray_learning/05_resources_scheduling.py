"""
==============================================================================
LESSON 5: RESOURCES AND SCHEDULING
==============================================================================

🎯 Learning Objectives:
    1. Understand Ray's resource model (CPUs, GPUs, memory)
    2. Specify resource requirements for tasks and actors
    3. Use custom resources for specialized hardware
    4. Control scheduling with placement groups
    5. Implement scheduling strategies
    6. Understand autoscaling triggers

⏱️ Estimated time: 40-50 minutes

Prerequisites: Complete lessons 01-04 first

Run this file: python 05_resources_scheduling.py
==============================================================================
"""

import ray
import time
import os

ray.init(ignore_reinit_error=True)

# ============================================================================
# SECTION 1: THE RESOURCE MODEL
# ============================================================================
"""
Ray's scheduling is RESOURCE-BASED:

1. Each node advertises resources (CPUs, GPUs, memory, custom)
2. Tasks/actors REQUEST resources
3. Scheduler places work where resources are available
4. Resources are RESERVED during execution

Key resources:
- CPU: Logical CPU cores
- GPU: GPU devices
- memory: Heap memory for the task
- object_store_memory: Space in object store
- Custom: Any string (e.g., "TPU", "special_hardware")

IMPORTANT: Resource requests are LOGICAL, not enforced!
- Requesting 4 CPUs doesn't limit you to 4 cores
- It's a scheduling hint, not a constraint
"""

print("=" * 60)
print("SECTION 1: The Resource Model")
print("=" * 60)

# Check available resources
resources = ray.cluster_resources()
print(f"\n📊 Cluster Resources:")
for key, value in sorted(resources.items()):
    if key in ["CPU", "GPU", "memory", "object_store_memory"]:
        if "memory" in key:
            print(f"   {key}: {value / 1e9:.2f} GB")
        else:
            print(f"   {key}: {value}")

# Available vs Used
available = ray.available_resources()
print(f"\n📊 Currently Available:")
print(f"   CPUs: {available.get('CPU', 0):.0f} / {resources.get('CPU', 0):.0f}")
print(f"   GPUs: {available.get('GPU', 0):.0f} / {resources.get('GPU', 0):.0f}")


# ============================================================================
# SECTION 2: SPECIFYING CPU REQUIREMENTS
# ============================================================================
"""
By default, each task uses 1 CPU.
You can change this with num_cpus parameter.

@ray.remote(num_cpus=N)  # Request N CPUs

Use cases:
- num_cpus=0: I/O tasks (don't need CPU scheduling)
- num_cpus=1: Default, single-threaded tasks
- num_cpus=4: Multi-threaded computation
- num_cpus=0.5: Fractional (2 tasks can share 1 CPU slot)
"""

print("\n" + "=" * 60)
print("SECTION 2: CPU Requirements")
print("=" * 60)

@ray.remote
def default_task():
    """Default: uses 1 CPU."""
    time.sleep(0.5)
    return f"Default task on PID {os.getpid()}"

@ray.remote(num_cpus=2)
def cpu_heavy_task():
    """Requests 2 CPUs (e.g., for parallel numpy)."""
    time.sleep(0.5)
    return f"CPU-heavy task (2 CPUs) on PID {os.getpid()}"

@ray.remote(num_cpus=0)
def io_task():
    """Requests 0 CPUs (pure I/O, doesn't need CPU slot)."""
    time.sleep(0.5)
    return f"I/O task (0 CPUs) on PID {os.getpid()}"

print("\n🔧 Running tasks with different CPU requirements:")

# Get current available CPUs
avail_cpus = ray.available_resources().get("CPU", 0)
print(f"   Available CPUs before: {avail_cpus:.0f}")

# Run default task
future = default_task.remote()
time.sleep(0.1)
avail_cpus = ray.available_resources().get("CPU", 0)
print(f"   During default_task (1 CPU): {avail_cpus:.0f} CPUs available")
ray.get(future)

# Run CPU-heavy task
future = cpu_heavy_task.remote()
time.sleep(0.1)
avail_cpus = ray.available_resources().get("CPU", 0)
print(f"   During cpu_heavy_task (2 CPUs): {avail_cpus:.0f} CPUs available")
ray.get(future)

# Run I/O task - doesn't consume CPU resources
future = io_task.remote()
time.sleep(0.1)
avail_cpus = ray.available_resources().get("CPU", 0)
print(f"   During io_task (0 CPUs): {avail_cpus:.0f} CPUs available")
ray.get(future)


# ============================================================================
# SECTION 3: GPU REQUIREMENTS
# ============================================================================
"""
Request GPUs with num_gpus parameter.

@ray.remote(num_gpus=N)  # Request N GPUs

Key behaviors:
- Ray sets CUDA_VISIBLE_DEVICES automatically
- Each GPU is assigned exclusively by default
- Fractional GPUs: num_gpus=0.5 allows 2 tasks per GPU

Check assigned GPUs inside task:
    os.environ.get("CUDA_VISIBLE_DEVICES")
"""

print("\n" + "=" * 60)
print("SECTION 3: GPU Requirements")
print("=" * 60)

@ray.remote(num_gpus=1)
def gpu_task():
    """Request 1 GPU."""
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "none")
    return f"GPU task - CUDA_VISIBLE_DEVICES={cuda_devices}"

@ray.remote(num_gpus=0.5)
def fractional_gpu_task(task_id):
    """Request half a GPU (2 tasks can share)."""
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "none")
    time.sleep(0.5)
    return f"Task {task_id} - CUDA_VISIBLE_DEVICES={cuda_devices}"

available_gpus = ray.cluster_resources().get("GPU", 0)
print(f"\n🎮 Available GPUs: {available_gpus:.0f}")

if available_gpus > 0:
    # Run GPU task
    result = ray.get(gpu_task.remote())
    print(f"   {result}")
    
    # Run fractional GPU tasks (can run 2 at once on 1 GPU)
    print("\n   Running 2 fractional GPU tasks (0.5 GPU each):")
    futures = [fractional_gpu_task.remote(i) for i in range(2)]
    for result in ray.get(futures):
        print(f"   {result}")
else:
    print("   No GPUs available - skipping GPU examples")
    print("   (On a GPU machine, tasks would be scheduled on GPUs)")


# ============================================================================
# SECTION 4: MEMORY REQUIREMENTS
# ============================================================================
"""
Specify memory requirements with memory parameter.

@ray.remote(memory=N)  # Request N bytes

Memory is used for:
- Scheduling decisions (task placed where memory available)
- NOT enforced as a hard limit (task can use more)

Best practices:
- Set memory requirements for memory-intensive tasks
- Helps scheduler make better placement decisions
- Use with autoscaler to provision right-sized nodes
"""

print("\n" + "=" * 60)
print("SECTION 4: Memory Requirements")
print("=" * 60)

@ray.remote(memory=500 * 1024 * 1024)  # 500 MB
def memory_intensive_task():
    """Task that needs 500 MB of memory."""
    import numpy as np
    # Allocate ~400 MB
    data = np.ones((50_000_000,), dtype=np.float64)  # 400 MB
    return f"Allocated {data.nbytes / 1e6:.0f} MB"

print("\n💾 Memory requirements:")
print("   memory=500*1024*1024  # Request 500 MB")

result = ray.get(memory_intensive_task.remote())
print(f"   Task result: {result}")


# ============================================================================
# SECTION 5: CUSTOM RESOURCES
# ============================================================================
"""
Define your own resources for specialized scheduling!

Use cases:
- Special hardware (TPUs, FPGAs)
- Software licenses (limited licenses)
- Logical grouping (team_a, team_b)
- Node-specific resources (has_ssd, has_fast_network)

Start Ray with custom resources:
    ray start --resources='{"TPU": 4, "special_hardware": 1}'
    
Or in Python:
    ray.init(resources={"TPU": 4})
"""

print("\n" + "=" * 60)
print("SECTION 5: Custom Resources")
print("=" * 60)

# Restart Ray with custom resources for demo
ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    resources={"special_hardware": 2, "software_license": 1}
)

print("\n🔧 Cluster restarted with custom resources:")
resources = ray.cluster_resources()
for key, value in sorted(resources.items()):
    if key not in ["memory", "object_store_memory", "node:__internal_head__"]:
        print(f"   {key}: {value}")

@ray.remote(resources={"special_hardware": 1})
def task_needs_special_hardware():
    """Task that requires special_hardware resource."""
    return "Running on special hardware!"

@ray.remote(resources={"software_license": 1})
def task_needs_license():
    """Task that requires a software license."""
    time.sleep(1)
    return "Used the software license!"

print("\n🚀 Running task with custom resource requirements:")
result = ray.get(task_needs_special_hardware.remote())
print(f"   {result}")

# License-limited parallelism
print("\n🔒 License-limited tasks (only 1 license available):")
print("   Starting 3 tasks, but only 1 can run at a time...")
start = time.time()
futures = [task_needs_license.remote() for _ in range(3)]
results = ray.get(futures)
print(f"   Completed in {time.time() - start:.1f}s (3 tasks × 1s, sequential due to 1 license)")


# ============================================================================
# SECTION 6: ACTOR RESOURCE REQUIREMENTS
# ============================================================================
"""
Actors can also specify resource requirements.

Resources are RESERVED for the actor's lifetime!

@ray.remote(num_cpus=2, num_gpus=1)
class MyActor:
    ...

Important: Actor resources are held until the actor is killed.
"""

print("\n" + "=" * 60)
print("SECTION 6: Actor Resources")
print("=" * 60)

@ray.remote(num_cpus=2)
class CPUIntensiveActor:
    """Actor that reserves 2 CPUs for its lifetime."""
    
    def __init__(self, name):
        self.name = name
    
    def process(self):
        return f"Actor {self.name} processing (2 CPUs reserved)"

print("\n🎭 Creating actor with 2 CPU requirement:")
avail_before = ray.available_resources().get("CPU", 0)
print(f"   Available CPUs before: {avail_before:.0f}")

actor = CPUIntensiveActor.remote("worker1")
time.sleep(0.5)  # Give actor time to start

avail_after = ray.available_resources().get("CPU", 0)
print(f"   Available CPUs after actor creation: {avail_after:.0f}")
print(f"   (2 CPUs reserved for actor's lifetime)")

result = ray.get(actor.process.remote())
print(f"   {result}")

# Kill actor to release resources
ray.kill(actor)
time.sleep(0.5)
avail_final = ray.available_resources().get("CPU", 0)
print(f"   Available CPUs after actor killed: {avail_final:.0f}")


# ============================================================================
# SECTION 7: SCHEDULING STRATEGIES
# ============================================================================
"""
Control WHERE tasks/actors are placed with scheduling strategies.

Built-in strategies:
1. DEFAULT: Prefer locality (data-aware), then load balance
2. SPREAD: Spread across nodes evenly
3. NodeAffinitySchedulingStrategy: Pin to specific node
4. PlacementGroupSchedulingStrategy: Use placement groups
"""

print("\n" + "=" * 60)
print("SECTION 7: Scheduling Strategies")
print("=" * 60)

from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)

@ray.remote
def get_node_id():
    """Get the node ID where this task runs."""
    return ray.get_runtime_context().get_node_id()

print("\n📍 Scheduling strategies:")

# Default strategy (locality-aware)
print("\n   1. DEFAULT (locality-aware, load balanced):")
refs = [get_node_id.remote() for _ in range(4)]
node_ids = ray.get(refs)
print(f"      Tasks placed on nodes: {set(node_ids)}")

# SPREAD strategy
print("\n   2. SPREAD (distribute across nodes):")
refs = [
    get_node_id.options(scheduling_strategy="SPREAD").remote()
    for _ in range(4)
]
node_ids = ray.get(refs)
print(f"      Tasks placed on nodes: {set(node_ids)}")
print("      (In a multi-node cluster, would see multiple node IDs)")


# ============================================================================
# SECTION 8: PLACEMENT GROUPS (GANG SCHEDULING)
# ============================================================================
"""
Placement Groups enable GANG SCHEDULING - ensure resources
are allocated together across nodes.

Use cases:
- Distributed training: All workers must start together
- Actor ensembles: Co-locate related actors
- Multi-GPU tasks: Reserve GPUs on specific nodes

Placement strategies:
- STRICT_PACK: All bundles on same node (if possible)
- PACK: Pack as tightly as possible
- STRICT_SPREAD: One bundle per node (must have enough nodes)
- SPREAD: Best-effort spread across nodes
"""

print("\n" + "=" * 60)
print("SECTION 8: Placement Groups")
print("=" * 60)

from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)

# Create a placement group with 2 bundles
# Each bundle requests 1 CPU
print("\n📦 Creating placement group with 2 bundles (1 CPU each):")

bundles = [
    {"CPU": 1},  # Bundle 0
    {"CPU": 1},  # Bundle 1
]

pg = placement_group(bundles, strategy="PACK")
ray.get(pg.ready())  # Wait for resources to be reserved

print(f"   Placement group created: {pg}")
print(f"   Bundles: {bundles}")

# Schedule tasks on specific bundles
@ray.remote(num_cpus=1)
def task_on_bundle(bundle_idx):
    return f"Task on bundle {bundle_idx}, node {ray.get_runtime_context().get_node_id()}"

print("\n🎯 Scheduling tasks on specific bundles:")

strategy = PlacementGroupSchedulingStrategy(
    placement_group=pg,
    placement_group_bundle_index=0
)
result0 = ray.get(get_node_id.options(
    scheduling_strategy=strategy,
    num_cpus=1
).remote())

strategy = PlacementGroupSchedulingStrategy(
    placement_group=pg,
    placement_group_bundle_index=1
)
result1 = ray.get(get_node_id.options(
    scheduling_strategy=strategy,
    num_cpus=1
).remote())

print(f"   Bundle 0 task on node: {result0}")
print(f"   Bundle 1 task on node: {result1}")
print("   (With PACK strategy, both should be on same node)")

# Cleanup
remove_placement_group(pg)
print("\n   Placement group removed, resources released")


# ============================================================================
# SECTION 9: PRACTICAL EXAMPLE - DISTRIBUTED TRAINING SETUP
# ============================================================================
"""
Real-world pattern: Setting up distributed training with placement groups.

Requirements:
- 4 workers for data parallel training
- Each worker needs 1 GPU
- Workers must be co-located on same node (or spread evenly)
"""

print("\n" + "=" * 60)
print("SECTION 9: Distributed Training Setup Pattern")
print("=" * 60)

print("""
🎓 Distributed Training with Placement Groups:

# Define resource bundles (one per worker)
bundles = [{"CPU": 1, "GPU": 1} for _ in range(4)]

# Create placement group
pg = placement_group(bundles, strategy="PACK")  # All on same node
# or strategy="SPREAD" for one per node

# Wait for resources
ray.get(pg.ready())

# Create training workers on specific bundles
@ray.remote(num_cpus=1, num_gpus=1)
class TrainingWorker:
    def __init__(self, rank):
        self.rank = rank
    
    def train(self, data):
        # Training logic
        return {"rank": self.rank, "loss": 0.1}

workers = [
    TrainingWorker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=i
        )
    ).remote(rank=i)
    for i in range(4)
]

# All workers are now guaranteed to have their resources!
""")


# ============================================================================
# SECTION 10: AUTOSCALING
# ============================================================================
"""
Ray Autoscaler automatically provisions/deprovisions nodes based on load.

How it works:
1. Monitor PENDING tasks/actors (tasks waiting for resources)
2. If resources needed > available, scale UP
3. If nodes idle for timeout period, scale DOWN

Key insight: Autoscaler reacts to RESOURCE REQUESTS, not utilization!

Configuration (in cluster.yaml):
available_node_types:
  worker:
    min_workers: 0
    max_workers: 10
    node_config: {...}
    resources: {"CPU": 4, "GPU": 1}

upscaling_speed: 1.0  # How fast to add nodes
idle_timeout_minutes: 5  # Time before scaling down
"""

print("\n" + "=" * 60)
print("SECTION 10: Autoscaling")
print("=" * 60)

print("""
📈 Ray Autoscaler:

   Triggered by PENDING tasks/actors:
   ┌─────────────────────────────────────────────────────┐
   │  Tasks Submitted → Resources Needed > Available     │
   │                    ↓                                │
   │  Autoscaler requests new nodes from cloud provider  │
   │                    ↓                                │
   │  New nodes join cluster with fresh resources        │
   │                    ↓                                │
   │  Pending tasks can now be scheduled                 │
   └─────────────────────────────────────────────────────┘

   Scale-down:
   ┌─────────────────────────────────────────────────────┐
   │  Node idle (no tasks/actors) for idle_timeout       │
   │                    ↓                                │
   │  Autoscaler terminates the node                     │
   └─────────────────────────────────────────────────────┘

   Check autoscaler status:
   ray status

   Force scaling demand (testing):
   ray.autoscaler.sdk.request_resources(num_cpus=100)
""")


# ============================================================================
# EXERCISES
# ============================================================================
"""
🏋️ EXERCISES - Master resource scheduling!

EXERCISE 1: Resource Contention
    Create 10 tasks, each requiring 2 CPUs.
    On a 4-CPU machine, observe:
    - How many run in parallel?
    - Total execution time vs. sequential

EXERCISE 2: Priority with Resources
    Create tasks with different resource requirements:
    - 5 tasks with num_cpus=1
    - 2 tasks with num_cpus=4
    Observe execution order and parallelism.

EXERCISE 3: GPU Memory Fraction
    Create an actor that simulates GPU memory usage.
    Use fractional GPUs (0.25, 0.5) to share a GPU.
    Track how many actors can coexist on one GPU.

EXERCISE 4: Custom Resource Semaphore
    Use custom resources to implement a semaphore:
    - Create resource "permits": 3
    - Tasks that acquire a permit (requires permits: 1)
    - Only 3 tasks can run concurrently

EXERCISE 5: Placement Group for ML
    Create a placement group that reserves:
    - 4 CPUs for data loading
    - 2 GPUs for training
    - 2 CPUs for logging
    All on the same node (STRICT_PACK).
"""

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
🏋️ Complete the exercises to master resource scheduling!

📝 Key takeaways:
   1. Resources (CPU, GPU, memory) are LOGICAL scheduling hints
   2. Custom resources enable specialized scheduling
   3. Actor resources are RESERVED for the actor's lifetime
   4. Scheduling strategies: DEFAULT, SPREAD, NodeAffinity
   5. Placement groups enable gang scheduling (all-or-nothing)
   6. Autoscaler reacts to PENDING tasks, not utilization

🔜 Next lesson: 06_fault_tolerance.py
   - Task retries and error handling
   - Actor restart policies
   - Checkpointing patterns
""")
