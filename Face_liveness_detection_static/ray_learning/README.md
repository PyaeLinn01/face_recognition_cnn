# 🚀 Ray Core Hands-On Learning Module

Welcome! This module teaches **Ray Core** from the ground up through executable experiments.
Ray Core is the foundation for all Ray libraries (Train, Tune, Serve, Data).

## 📋 Prerequisites

```bash
pip install ray[default]
```

## 🗂️ Lesson Structure

| Lesson | File | Concepts |
|--------|------|----------|
| 1 | `01_basics_tasks.py` | `ray.init()`, `@ray.remote`, `.remote()`, `ray.get()` |
| 2 | `02_parallel_patterns.py` | Map-reduce, task dependencies, futures, `ray.wait()` |
| 3 | `03_actors_stateful.py` | Stateful workers, actor handles, method calls |
| 4 | `04_object_store.py` | `ray.put()`, object references, zero-copy, data passing |
| 5 | `05_resources_scheduling.py` | CPU/GPU allocation, custom resources, placement |
| 6 | `06_fault_tolerance.py` | Task retries, actor restarts, error handling |

## 🎯 How to Learn

1. **Read the code** - Each file has detailed comments explaining concepts
2. **Run it** - Execute each file and observe the output
3. **Experiment** - Modify parameters, add print statements, break things!
4. **Challenge yourself** - Each file ends with exercises

## 🧠 Core Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│                      Ray Cluster                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Worker 1   │    │  Worker 2   │    │  Worker N   │     │
│  │  ┌───────┐  │    │  ┌───────┐  │    │  ┌───────┐  │     │
│  │  │ Task  │  │    │  │ Actor │  │    │  │ Task  │  │     │
│  │  └───────┘  │    │  └───────┘  │    │  └───────┘  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│              ┌─────────────────────────┐                    │
│              │   Distributed Object    │                    │
│              │        Store            │                    │
│              │  (Shared Memory)        │                    │
│              └─────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## 🔑 The Three Primitives

1. **Tasks** = Stateless functions (`@ray.remote` on functions)
2. **Actors** = Stateful classes (`@ray.remote` on classes)  
3. **Objects** = Immutable data in distributed shared memory

Master these three, and you understand Ray Core!

## 📊 Ray Dashboard

When you run `ray.init()`, a dashboard starts at http://127.0.0.1:8265
Use it to monitor tasks, actors, memory, and logs.

---

**Start with:** `python 01_basics_tasks.py`
