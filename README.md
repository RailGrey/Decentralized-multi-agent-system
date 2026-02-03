# Multi-Agent RAG System for Competitive Programming

A multi-agent system that uses Retrieval-Augmented Generation (RAG) and collaborative problem-solving to tackle coding challenges. Agents specialize in different algorithmic domains, share knowledge through brainstorming, peer-review each other's solutions, and vote on the best approach. There is no moderator in this system.

---

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [RAG Agent Architecture](#rag-agent-architecture)
- [Brainstorm Solver Pipeline](#brainstorm-solver-pipeline)

---

## Introduction

This project implements a **multi-agent system** where specialized AI agents collaborate to solve competitive programming problems. Each agent is an expert in a specific algorithmic domain (e.g., dynamic programming, graphs, greedy algorithms) and uses RAG to learn from a curated knowledge base of problems and solutions.

### Key Features

- **🤖 Specialized Agents**: Each agent focuses on a specific topic (DP, graphs, trees, etc.)
- **📚 RAG-Powered**: Agents retrieve relevant examples from a vector database before solving
- **💡 Collaborative Brainstorming**: Agents share ideas and build consensus
- **🔍 Peer Review**: Multi-round code review and revision process
- **🗳️ Democratic Voting**: Agents vote to select the best solution
- **🎯 Production-Ready**: Uses Qdrant for vectors, Mistral API for LLM, with caching

---

## Prerequisites

### Required

- **Python 3.12+**
- **Mistral API Key** ([Get one here](https://console.mistral.ai/), it has free usage plans)
- **HF token** (optional, for fast download of dataset)

### Dependencies

```bash
poetry install
```

**Core libraries:**
- `mistralai (>=1.11.1,<2.0.0)` - Mistral API client
- `qdrant-client (>=1.16.2,<2.0.0)` - Vector database
- `datasets (>=4.5.0,<5.0.0)` - HF library to download dataset
- `dotenv (>=0.9.9,<0.10.0)` - Crentials managing

---

## Installation

### Step 1: Clone & Install

```bash
# Clone repository
git clone <your-repo-url>
cd decentralized-multi-agent-system

# Install dependencies
poetry install
```

Create credentials file (in env directory) and insert MISTRAL_API_KEY

### Step 2: Build Knowledge Base (One-Time)

Download dataset using download_dataset.py script. This creates `dataset` folder where it saves.

I've used [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests).

### Step 3: Run the System

You can run it as a script using main.py, or do it step-by-step in demo.ipynb notebook

## Project Structure

```
multi-agent-rag-system/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── Core Components
│   ├── rag_agent.py                  # RAG agent implementation
│   ├── brainstorm_solver.py          # Multi-agent orchestration
│   └── summarizer_agent.py           # Final summary generation
│
├── Utilities
│   ├── download_dataset.py           # Download dataset
│   ├── knowledge_builder.py          # Extract & save problems
│   └── main.py                       # Usage examples
│
├── Data Directories (generated)
│   ├── agent_knowledge/               # Problem datasets (don't commit to git)
│   │   ├── math_agent.json
│   │   ├── greedy_agent.json
│   │   ├── dp_agent.json
│   │   └── ...
│   │
│   ├── qdrant_storage/                # Vector database (don't commit)
│   |   └── <agent>/
│   |       └── *.json
│   │
│   └── embedding_cache/               # API response cache (don't commit)
│       └── <agent>/
│           └── *.json
│
├── settings/                         # Configuration management
│   ├── agent_config.py              # Agent configurations
│   ├── prompts.py                   # All system prompt templates
│   ├── schemas.py                   # Schemas for structured output
│   ├── logger.py                    # Logging configuration
│
├── env/                             # Environment configuration
│   ├── __init__.py
│   ├── credentials.env              # Production credentials (gitignored)
│   ├── demo_credentials.env         # Demo credentials
```

---

## RAG Agent Architecture

### Overview

Each `MistralRAGAgent` is a self-contained expert that:
1. Stores knowledge in **Qdrant** (vector database)
2. Retrieves relevant examples when given a task
3. Uses **Mistral API** to generate solutions
4. Caches embeddings to minimize API costs

### Architecture Diagram

```
┌─────────────────────────────────────────────┐
│          MistralRAGAgent                    │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌──────────────┐    │
│  │   Mistral    │      │   Qdrant     │    │
│  │   API        │◄────►│   Vector DB  │    │
│  │              │      │              │    │
│  │ • Embeddings │      │ • 1024-dim   │    │
│  │ • Chat       │      │ • COSINE     │    │
│  └──────────────┘      │ • HNSW Index │    │
│         │              └──────────────┘    │
│         │                      ▲           │
│         ▼                      │           │
│  ┌──────────────┐      ┌──────────────┐   │
│  │  Embedding   │      │  Knowledge   │   │
│  │  Cache       │      │  Base        │   │
│  │  (Disk)      │      │  (Problems)  │   │
│  └──────────────┘      └──────────────┘   │
│                                             │
└─────────────────────────────────────────────┘
```

### Component Details

#### 1. **Knowledge Storage (Qdrant)**

Each agent stores problems in its own Qdrant collection:

```python
Collection: "math_agent_knowledge"
├── Point 1:
│   ├── Vector: [0.123, -0.456, ...] (1024-dim)
│   └── Payload:
│       ├── content: "Problem description"
│       ├── solution: "def solve()..."
│       ├── cf_tag: "math"
│       └── problem_id: 1
```

**Benefits:**
- **Fast retrieval**: O(log n) vs O(n) linear search
- **Scalable**: Handles millions of vectors

#### 2. **Embedding Cache**

To avoid redundant API calls, embeddings are cached to disk:

```
embedding_cache/Math agent/
├── a3f5e2c1b4d9.json  # SHA256(text) → embedding
└── f7e8d2c3a1b6.json
```

**Cache file format:**
```json
{
  "text": "original text",
  "embedding": [0.123, -0.456, ...],
  "metadata": {...},
  "model": "mistral-embed"
}
```

#### 3. **Retrieval Process**

```python
# 1. User query
query = "Find shortest path"

# 2. Check cache for query embedding
embedding = agent._get_embedding(query)  # Cache hit or API call

# 3. Search Qdrant
results = qdrant.query_points(
    collection="graph_agent_knowledge",
    query=query_embedding,
    limit=top_k
)

# 4. Format context with problems AND solutions
context = """
[1] Problem:
Given a weighted graph, find shortest path...

Solution:
def dijkstra(graph, start):
    ...
"""

# 5. Send to Mistral API with context
response = mistral.chat.complete(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context}\n\n{query}"}
    ]
)
```

#### 4. **Single Agent Solve**

For standalone usage (not in multi-agent system):

```python
result = agent.solve("Find longest palindrome substring")

# Returns:
{
    "solution": "def longest_palindrome(s):\n    ...",
    "description": "Use dynamic programming with 2D table...",
    "confidence": 0.85
}
```

---

## Brainstorm Solver Pipeline

The `BrainstormSolver` orchestrates multiple agents through a **7-phase pipeline** that produces high-quality solutions through collaboration, peer review, and democratic selection.

### Pipeline Overview

```
Phase 1: Confidence Estimation
    ↓
Phase 2: Core Agent Selection (mean threshold)
    ↓
Phase 3: Brainstorm (collaborative ideation)
    ↓
Phase 4: Code Generation
    ↓
Phase 5: Peer Review & Revision
    ↓
Phase 6: Voting
    ↓
Phase 7: Final Summary
```

---

### Phase 1: Confidence Estimation

**Goal:** Each agent estimates how confident they are in solving the task.

**Process:**
1. Every agent receives the task
2. Retrieves relevant knowledge from its Qdrant collection
3. Self-reports confidence (0.0 - 1.0)

**Schema:**
```json
{
  "confidence": 0.85,
  "reason": "This involves graph traversal which is my expertise"
}
```

**Output Example:**
```
PHASE 1 — Confidence Estimation
======================================================================
  Math Agent             0.25  — Not a mathematical problem
  Graph Agent            0.90  — Graph algorithms are my specialty
  DP Agent               0.40  — Could use DP but not optimal
  Tree Agent             0.30  — Trees are special graphs but not core here
```

**Key Implementation:**
- Uses RAG to inform confidence (checks knowledge base first)
- Robust parsing handles text responses ("high" → 0.85)
- Fallback to 0.5 if parsing fails

**Why does it works:**

Of course, agents could be overconfident in their abialities and generate high scores. However, system uses agents with the same llm behind, which means that bias in estimating confidence is equal for all agents.

---

### Phase 2: Core Agent Selection

**Goal:** Filter to agents with above-average confidence.

**Algorithm:**
```python
threshold = mean(all_confidences)
core_agents = [agent for agent in agents if confidence >= threshold]
core_agents.sort(by=confidence, descending=True)
```

**Output Example:**
```
PHASE 2 — Core Agent Selection (threshold = mean)
======================================================================
  Mean threshold : 0.46
  Core agents    : ['Graph Agent', 'DP Agent']
```

**Why mean threshold?**
- Adaptive: automatically adjusts to task difficulty
- Fair: doesn't rely on arbitrary cutoffs

**Result:** Only high-confidence agents proceed to brainstorming.

---

### Phase 3: Brainstorm

**Goal:** Agents collaboratively generate ideas, building a shared knowledge base.

**Process:**
1. Agents called sequentially (highest confidence first)
2. Each agent can:
   - **Add** a new idea
   - **Support** an existing idea (adds their confidence to its score)
   - **Criticize** an idea (flags flaws)
3. Ideas are scored: `score = Σ(confidence of all supporters)`
4. Ideas re-sorted after each contribution

**Schema:**
```json
{
  "contributions": [
    {
      "action": "add",
      "idea_id": "",
      "text": "Use Dijkstra's algorithm with priority queue",
      "reason": ""
    },
    {
      "action": "support",
      "idea_id": "idea_1",
      "text": "",
      "reason": ""
    },
    {
      "action": "criticize",
      "idea_id": "idea_2",
      "text": "",
      "reason": "This doesn't handle negative weights"
    }
  ]
}
```

**Output Example:**
```
PHASE 3 — Brainstorm
======================================================================
  Graph Agent            ADD       [idea_1] Use Dijkstra with binary heap
  DP Agent               ADD       [idea_2] Dynamic programming on paths
  Graph Agent            CRITICIZE [idea_2] DP is overkill for this...
  DP Agent               SUPPORT   [idea_1] score → 1.30

Final ideas (sorted by score):
  [idea_1] score=1.30 (supporters: Graph Agent, DP Agent)
           Use Dijkstra with binary heap for O(N log N)
  
  [idea_2] score=0.40 (supporters: DP Agent)
           Dynamic programming on paths
           ⚠ criticized by Graph Agent: DP is overkill...
```

**Key Features:**
- **Append-only memory**: All ideas preserved
- **Scored, not voted out**: Weak ideas sink, don't disappear
- **Criticism is metadata**: Agents see criticisms but make their own judgment
- **No duplicates**: Agents support existing ideas rather than re-proposing

---

### Phase 4: Code Generation

**Goal:** Each core agent independently writes a complete solution.

**Process:**
1. Each agent receives:
   - Original task
   - Full brainstorm results (sorted by score)
   - Their own RAG-retrieved knowledge
2. Generates code referencing specific idea_ids

**Schema:**
```json
{
  "ideas_used": ["idea_1", "idea_3"],
  "solution": "def shortest_path(graph, start, end):\n    ...",
  "time_complexity": "O(N log N)",
  "space_complexity": "O(N)",
  "edge_cases": [
    "Disconnected graph - return infinity",
    "Negative weights - not supported",
    "Single node - return 0"
  ]
}
```

**Output Example:**
```
PHASE 4 — Code Generation
======================================================================
  Graph Agent            ideas=['idea_1']  time=O(N log N)
  DP Agent               ideas=['idea_1', 'idea_2']  time=O(V * E)
```

**Key Features:**
- Independent generation (no collaboration here)
- Grounded in brainstorm (must reference idea_ids)
- Explicit complexity analysis required
- Edge case handling required

---

### Phase 5: Peer Review & Revision

**Goal:** Iteratively improve solutions through peer review.

**Process (per round):**
1. **Review Phase:**
   - Every agent reviews every OTHER agent's code
   - Verdict: "approve" or "fix"
   - Provides feedback and counterexamples
2. **Revision Phase:**
   - Each agent revises their own code based on feedback
   - Explains changes (or why feedback was wrong)
3. **Convergence Check:**
   - If all verdicts are "approve" → stop early
   - Otherwise → next round (up to `max_review_rounds`)

**Review Schema:**
```json
{
  "reviews": [
    {
      "agent_reviewed": "DP Agent",
      "verdict": "fix",
      "feedback": "Time complexity claim is incorrect. It's O(V^3) not O(V*E)",
      "counterexample": "Graph with V=100, E=200 takes 1M operations not 20K"
    }
  ]
}
```

**Revision Schema:**
```json
{
  "ideas_used": ["idea_1"],
  "solution": "<revised code>",
  "time_complexity": "O(V^3)",
  "space_complexity": "O(V^2)",
  "edge_cases": ["..."],
  "changes_made": "Fixed complexity analysis. Changed algorithm to Floyd-Warshall"
}
```

**Output Example:**
```
PHASE 5 — Peer Review & Revision
======================================================================

  ── Review round 1/3 ──
    Graph Agent          → DP Agent              [fix]
    DP Agent             → Graph Agent           [approve]
    
    Graph Agent           revised — No changes, feedback was incorrect
    DP Agent              revised — Fixed complexity analysis per feedback
    
  ── Review round 2/3 ──
    Graph Agent          → DP Agent              [approve]
    DP Agent             → Graph Agent           [approve]
    
  ✓ All solutions approved — stopping early.
```

**Why this works:**
- **Redundancy**: Multiple expert perspectives catch bugs
- **Iterative refinement**: Solutions improve over rounds
- **Peer pressure**: Agents justify their decisions
- **No moderator**: Consensus emerges organically

---

### Phase 6: Voting

**Goal:** Democratic selection of the best solution.

**Process:**
1. Each agent votes on ALL solutions (including their own)
2. Scores range from 0.0 to 10.0
3. Criteria: correctness, efficiency, clarity, edge cases
4. Winner = highest total score
5. Tiebreaker: random selection

**Schema:**
```json
{
  "votes": [
    {
      "agent_name": "Graph Agent",
      "score": 9.0,
      "reasoning": "Optimal time complexity, handles all edge cases, clean code"
    },
    {
      "agent_name": "DP Agent",
      "score": 7.5,
      "reasoning": "Correct but suboptimal complexity"
    }
  ]
}
```

**Output Example:**
```
PHASE 6 — Voting
======================================================================
  Math Agent             cast 2 votes
  Graph Agent            cast 2 votes
  DP Agent               cast 2 votes

  Winner: Graph Agent with total score 26.50

  Final scores:
    Graph Agent            26.50
    DP Agent               19.20
```

**Tiebreaker:**
```python
if len(winners) > 1:
    winner = random.choice(winners)
    print(f"Tie between {winners}, random selection: {winner}")
```

**Why voting?**
- Democratic: no single agent decides
- Objective: scored on concrete criteria
- Transparent: full vote breakdown visible
- Fair: includes self-voting (forces honesty)

---

### Phase 7: Final Summary

**Goal:** Professional summary of the winning solution.

**Process:**
1. `MistralSummarizerAgent` receives:
   - Winning solution
   - All other solutions (for comparison)
   - Complete voting results
   - Brainstorm ideas
2. Produces structured summary

**Schema:**
```json
{
  "final_solution": "<winning solution, potentially enhanced>",
  "confidence": 0.88,
  "winner_rationale": "Graph Agent's Dijkstra implementation was praised for...",
  "key_strengths": [
    "Optimal O(N log N) time complexity",
    "Comprehensive edge case handling",
    "Clean, readable implementation"
  ],
  "potential_improvements": [
    "Could add path reconstruction as DP Agent suggested"
  ],
  "voting_consensus": "Strong consensus - received 8.5-9.0 from all voters",
  "brainstorm_impact": "The priority queue optimization from idea_1 was crucial"
}
```

**Output Example:**
```
PHASE 7 — Final Summary
======================================================================
  Summarizer created final summary
  Winner solution: Graph Agent
  Confidence: 0.88

WHY THIS SOLUTION WON:
----------------------------------------------------------------------
Graph Agent's solution received unanimous high scores (8.5-9.0) for its
elegant implementation of Dijkstra's algorithm with optimal time complexity.

FINAL SOLUTION:
----------------------------------------------------------------------
def shortest_path(graph, start, end):
    import heapq
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        if current == end:
            return current_dist
        ...

KEY STRENGTHS:
----------------------------------------------------------------------
1. Optimal O(N log N) time complexity using binary heap
2. Handles disconnected graphs correctly
3. Clean, production-ready code
4. Comprehensive edge case coverage

POTENTIAL IMPROVEMENTS:
----------------------------------------------------------------------
1. Could add path reconstruction (store predecessors)
2. DP Agent's memoization approach could work for specific graph types

VOTING CONSENSUS:
----------------------------------------------------------------------
Strong consensus - all agents gave 8.5+ scores with specific praise for
the priority queue optimization and edge case handling.

BRAINSTORM IMPACT:
----------------------------------------------------------------------
The brainstorm phase was crucial - idea_1 (Dijkstra with binary heap)
was supported by both Graph and DP agents early on, which guided the
final implementation toward the optimal approach.
```

---