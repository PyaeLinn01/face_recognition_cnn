"""
==============================================================================
LESSON 2: PARALLEL PATTERNS
==============================================================================

🎯 Learning Objectives:
    1. Master the Map pattern (parallel apply)
    2. Implement Map-Reduce with Ray
    3. Chain tasks with dependencies (task graphs)
    4. Use ray.wait() for streaming results
    5. Handle task timeouts and partial results

⏱️ Estimated time: 30-40 minutes

Prerequisites: Complete 01_basics_tasks.py first

Run this file: python 02_parallel_patterns.py
==============================================================================
"""

import ray
import time
import random

ray.init(ignore_reinit_error=True)

# ============================================================================
# SECTION 1: THE MAP PATTERN
# ============================================================================
"""
Map Pattern: Apply the same function to every element in parallel.

    [1, 2, 3, 4, 5]
         │
         ▼ map(square)
    [1, 4, 9, 16, 25]

This is the most common parallel pattern. Perfect for:
- Batch inference (apply model to many inputs)
- Data transformation (process each record)
- Embarrassingly parallel problems (no inter-task communication)
"""

print("=" * 60)
print("SECTION 1: The Map Pattern")
print("=" * 60)

@ray.remote
def process_item(item):
    """Simulate processing a single item."""
    time.sleep(0.5)  # Simulate work
    return item ** 2

# Map over a list in parallel
items = list(range(10))
print(f"\n📥 Input: {items}")

start = time.time()
# Launch all tasks in parallel (map phase)
futures = [process_item.remote(item) for item in items]
# Collect all results
results = ray.get(futures)
print(f"📤 Output: {results}")
print(f"⏱️  Time: {time.time() - start:.2f}s (10 items × 0.5s each, but parallel!)")


# ============================================================================
# SECTION 2: MAP-REDUCE PATTERN
# ============================================================================
"""
Map-Reduce: Map + Reduce (aggregate)

    Phase 1 (Map):     [a, b, c, d] → [f(a), f(b), f(c), f(d)]
    Phase 2 (Reduce):  [f(a), f(b), f(c), f(d)] → aggregate

Example: Word count across documents
    Map:    Each doc → {word: count}
    Reduce: Merge all dictionaries

Ray excels at this because:
1. Map phase runs fully parallel
2. Reduce can be hierarchical (tree reduction)
"""

print("\n" + "=" * 60)
print("SECTION 2: Map-Reduce Pattern")
print("=" * 60)

@ray.remote
def map_word_count(text):
    """Map phase: Count words in a single document."""
    word_counts = {}
    for word in text.lower().split():
        word = word.strip(".,!?")
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

@ray.remote
def reduce_word_counts(counts_list):
    """Reduce phase: Merge multiple word count dictionaries."""
    merged = {}
    for counts in counts_list:
        for word, count in counts.items():
            merged[word] = merged.get(word, 0) + count
    return merged

# Sample documents
documents = [
    "Ray is a distributed computing framework",
    "Ray makes parallel programming easy",
    "Distributed computing with Ray is powerful",
    "Python and Ray work well together",
    "Ray is used for machine learning",
]

print(f"\n📄 Processing {len(documents)} documents...")

# Map phase: Count words in each document (parallel)
map_futures = [map_word_count.remote(doc) for doc in documents]
word_counts = ray.get(map_futures)

print("\n   Map phase results (per document):")
for i, counts in enumerate(word_counts):
    print(f"   Doc {i}: {dict(list(counts.items())[:3])}...")

# Reduce phase: Merge all counts
final_counts = ray.get(reduce_word_counts.remote(word_counts))

print(f"\n   Reduce phase result (merged):")
# Sort by count and show top words
top_words = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"   Top 5 words: {top_words}")


# ============================================================================
# SECTION 3: TREE REDUCE (HIERARCHICAL AGGREGATION)
# ============================================================================
"""
For large-scale reduce operations, a single reducer becomes a bottleneck.
Tree Reduce solves this by reducing in layers:

    Level 0: [1, 2, 3, 4, 5, 6, 7, 8]
                  ↓
    Level 1: [3, 7, 11, 15]  (pairs reduced)
                  ↓
    Level 2: [10, 26]        (pairs reduced)
                  ↓
    Level 3: [36]            (final result)

This keeps parallelism at each level!
"""

print("\n" + "=" * 60)
print("SECTION 3: Tree Reduce")
print("=" * 60)

@ray.remote
def add_values(a, b):
    """Add two values (for tree reduction)."""
    time.sleep(0.2)  # Simulate work
    return a + b

def tree_reduce(values):
    """Recursively reduce values in a tree pattern."""
    # Base case: single value
    if len(values) == 1:
        return values[0]
    
    # Reduce pairs in parallel
    next_level = []
    for i in range(0, len(values), 2):
        if i + 1 < len(values):
            # Pair exists - reduce them
            next_level.append(add_values.remote(values[i], values[i + 1]))
        else:
            # Odd element - carry forward
            next_level.append(values[i])
    
    # Recursively reduce the next level
    return tree_reduce(next_level)

# Generate some values
values = list(range(1, 17))  # [1, 2, 3, ..., 16]
print(f"\n📊 Values: {values}")
print(f"   Sum should be: {sum(values)}")

start = time.time()
result_ref = tree_reduce(values)
result = ray.get(result_ref)
print(f"\n   Tree reduce result: {result}")
print(f"   Time: {time.time() - start:.2f}s")
print(f"   (Tree depth: 4 levels × 0.2s = ~0.8s, not 15 × 0.2s = 3s!)")


# ============================================================================
# SECTION 4: TASK DEPENDENCIES (DAG / TASK GRAPHS)
# ============================================================================
"""
Tasks can depend on other tasks by passing ObjectRefs as arguments.
Ray automatically waits for dependencies before executing.

    A ──→ C ──→ E
          ↑     ↑
    B ────┘     │
                │
    D ──────────┘

This creates a DAG (Directed Acyclic Graph) of tasks.
Ray's scheduler handles execution order automatically!
"""

print("\n" + "=" * 60)
print("SECTION 4: Task Dependencies (DAG)")
print("=" * 60)

@ray.remote
def load_data(source):
    """Stage 1: Load data from source."""
    time.sleep(0.5)
    print(f"   [load_data] Loaded from {source}")
    return {"source": source, "data": list(range(10))}

@ray.remote
def preprocess(data_ref):
    """Stage 2: Preprocess data (depends on load_data)."""
    time.sleep(0.3)
    data = data_ref  # Ray automatically resolves the ObjectRef!
    processed = [x * 2 for x in data["data"]]
    print(f"   [preprocess] Processed {len(processed)} items")
    return processed

@ray.remote
def train_model(data_ref):
    """Stage 3: Train model (depends on preprocess)."""
    time.sleep(0.4)
    data = data_ref
    model = {"weights": sum(data) / len(data)}  # Silly "model"
    print(f"   [train_model] Trained with avg weight: {model['weights']}")
    return model

@ray.remote
def evaluate(model_ref, test_data_ref):
    """Stage 4: Evaluate (depends on train_model AND test data)."""
    time.sleep(0.2)
    model = model_ref
    test_data = test_data_ref
    score = model["weights"] * len(test_data)
    print(f"   [evaluate] Score: {score}")
    return score

print("\n🔗 Building task dependency graph...")
print("""
    load_train ──→ preprocess ──→ train ──→ evaluate
                                      ↑         ↑
    load_test ─────────────────────────────────┘
""")

# Build the DAG - tasks won't execute until we call ray.get()
train_data = load_data.remote("train.csv")
test_data = load_data.remote("test.csv")  # Runs in parallel with train_data!

preprocessed = preprocess.remote(train_data)  # Waits for train_data
model = train_model.remote(preprocessed)       # Waits for preprocessed
score = evaluate.remote(model, test_data)      # Waits for BOTH model AND test_data

print("\n   Executing DAG...")
start = time.time()
final_score = ray.get(score)
print(f"\n   Final score: {final_score}")
print(f"   Total time: {time.time() - start:.2f}s")
print("   (Notice: load_train and load_test ran in parallel!)")


# ============================================================================
# SECTION 5: ray.wait() - STREAMING RESULTS
# ============================================================================
"""
Sometimes you don't want to wait for ALL tasks to complete.
ray.wait() lets you process results as they become available.

Use cases:
- Progress reporting
- Processing fastest results first
- Timeouts and cancellation
- Streaming pipelines
"""

print("\n" + "=" * 60)
print("SECTION 5: ray.wait() - Streaming Results")
print("=" * 60)

@ray.remote
def variable_time_task(task_id):
    """Task that takes random time to complete."""
    delay = random.uniform(0.1, 2.0)
    time.sleep(delay)
    return f"Task {task_id} (took {delay:.2f}s)"

# Launch tasks with varying completion times
print("\n🚀 Launching 6 tasks with random durations...")
futures = [variable_time_task.remote(i) for i in range(6)]

# Process results as they complete
print("\n   Processing results as they arrive:")
remaining = futures.copy()
completed_order = []

while remaining:
    # Wait for at least one task to complete
    # num_returns=1 means "give me the first completed task"
    done, remaining = ray.wait(remaining, num_returns=1)
    
    # Process the completed task
    result = ray.get(done[0])
    completed_order.append(result)
    print(f"   ✅ Completed: {result}")

print(f"\n   Completion order: {[r.split()[1] for r in completed_order]}")
print("   (Notice: faster tasks complete first, not submission order!)")


# ============================================================================
# SECTION 6: TIMEOUTS WITH ray.wait()
# ============================================================================
"""
ray.wait() also supports timeouts - crucial for production systems.

Parameters:
- num_returns: How many completed tasks to wait for
- timeout: Maximum seconds to wait (returns early if reached)
"""

print("\n" + "=" * 60)
print("SECTION 6: Timeouts with ray.wait()")
print("=" * 60)

@ray.remote
def slow_task(task_id, duration):
    """Task that takes a specific duration."""
    time.sleep(duration)
    return f"Task {task_id} done"

# Launch some slow tasks
print("\n⏰ Launching tasks with 1s, 2s, and 5s durations...")
futures = [
    slow_task.remote(1, 1.0),
    slow_task.remote(2, 2.0),
    slow_task.remote(3, 5.0),  # This one is slow!
]

# Wait with a timeout
print("   Waiting for 3 seconds max...")
done, not_done = ray.wait(futures, num_returns=3, timeout=3.0)

print(f"\n   Completed: {len(done)} tasks")
print(f"   Still running: {len(not_done)} tasks")

for ref in done:
    print(f"   ✅ {ray.get(ref)}")
for ref in not_done:
    print(f"   ⏳ Task still pending...")

# Get the remaining results (will block until complete)
print("\n   Waiting for remaining tasks...")
remaining_results = ray.get(not_done)
print(f"   ✅ {remaining_results}")


# ============================================================================
# SECTION 7: PRACTICAL PATTERN - PARALLEL WEB SCRAPER
# ============================================================================
"""
Let's combine patterns into a practical example:
A parallel web scraper that:
1. Maps fetch_url over URLs (parallel)
2. Uses ray.wait() for progress reporting
3. Handles timeouts for slow URLs
"""

print("\n" + "=" * 60)
print("SECTION 7: Practical Pattern - Parallel URL Fetcher")
print("=" * 60)

@ray.remote
def fetch_url(url):
    """Simulate fetching a URL."""
    # Simulate variable network latency
    latency = random.uniform(0.1, 1.5)
    time.sleep(latency)
    
    # Simulate occasional failures
    if random.random() < 0.1:
        raise Exception(f"Failed to fetch {url}")
    
    return {"url": url, "latency": latency, "size": random.randint(1000, 50000)}

# URLs to fetch
urls = [f"https://example.com/page{i}" for i in range(10)]

print(f"\n🌐 Fetching {len(urls)} URLs in parallel...")

# Launch all fetches
futures = {fetch_url.remote(url): url for url in urls}
remaining = list(futures.keys())
results = []
errors = []

# Process with progress reporting
start = time.time()
while remaining:
    done, remaining = ray.wait(remaining, num_returns=1, timeout=0.5)
    
    for ref in done:
        url = futures[ref]
        try:
            result = ray.get(ref)
            results.append(result)
            print(f"   ✅ {url}: {result['size']} bytes in {result['latency']:.2f}s")
        except Exception as e:
            errors.append(url)
            print(f"   ❌ {url}: {e}")
    
    if not done and remaining:
        print(f"   ⏳ Waiting... ({len(remaining)} remaining)")

print(f"\n📊 Summary:")
print(f"   Successful: {len(results)}")
print(f"   Failed: {len(errors)}")
print(f"   Total time: {time.time() - start:.2f}s")


# ============================================================================
# EXERCISES
# ============================================================================
"""
🏋️ EXERCISES - Practice parallel patterns!

EXERCISE 1: Parallel Prime Counter
    Use map-reduce to count primes in ranges:
    - Map: Each task counts primes in a range (e.g., 0-1000, 1000-2000)
    - Reduce: Sum the counts

EXERCISE 2: Image Processing Pipeline
    Create a DAG for image processing:
    - load_image() → resize() → apply_filter() → save()
    - Process multiple images with the same pipeline in parallel
    
EXERCISE 3: First N Completed
    Launch 20 tasks with random durations.
    Use ray.wait() to get only the 5 fastest results.
    Cancel (ignore) the rest.

EXERCISE 4: Progress Bar
    Implement a progress bar using ray.wait():
    - Launch 50 tasks
    - Use ray.wait() in a loop to update progress: [=====>    ] 50%
    
EXERCISE 5: Fault-Tolerant Scraper
    Enhance the web scraper to:
    - Retry failed URLs up to 3 times
    - Track retry count per URL
    - Report final success/failure statistics
"""

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
🏋️ Try the exercises above to master parallel patterns!

📝 Key takeaways:
   1. Map pattern: [f.remote(x) for x in items] + ray.get()
   2. Map-Reduce: Map phase → Reduce phase (parallel aggregation)
   3. Tree Reduce: Hierarchical reduction keeps parallelism
   4. Task DAGs: Pass ObjectRefs to create dependencies
   5. ray.wait(): Process results as they arrive, with timeouts

🔜 Next lesson: 03_actors_stateful.py
   - Stateful computation with actors
   - Actor handles and method calls
   - When to use actors vs tasks
""")
