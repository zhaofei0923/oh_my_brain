# Architecture

OH MY BRAIN follows a centralized coordinator pattern with distributed workers.

## System Overview

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                        BRAIN                            │
                    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
                    │  │   Context    │  │    Task      │  │    Model     │  │
                    │  │   Manager    │  │  Scheduler   │  │   Router     │  │
                    │  └──────────────┘  └──────────────┘  └──────────────┘  │
                    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
                    │  │    Doc       │  │   Safety     │  │     Git      │  │
                    │  │   Parser     │  │   Checker    │  │   Manager    │  │
                    │  └──────────────┘  └──────────────┘  └──────────────┘  │
                    │                         │                               │
                    │                    ZeroMQ ROUTER                        │
                    └─────────────────────────┼───────────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
    ┌─────────┴─────────┐         ┌─────────┴─────────┐         ┌─────────┴─────────┐
    │     WORKER 1      │         │     WORKER 2      │         │     WORKER 3      │
    │  ┌─────────────┐  │         │  ┌─────────────┐  │         │  ┌─────────────┐  │
    │  │ Mini-Agent  │  │         │  │ Mini-Agent  │  │         │  │ Mini-Agent  │  │
    │  │  Adapter    │  │         │  │  Adapter    │  │         │  │  Adapter    │  │
    │  └─────────────┘  │         │  └─────────────┘  │         │  └─────────────┘  │
    │  ┌─────────────┐  │         │  ┌─────────────┐  │         │  ┌─────────────┐  │
    │  │Brain Client │  │         │  │Brain Client │  │         │  │Brain Client │  │
    │  └─────────────┘  │         │  └─────────────┘  │         │  └─────────────┘  │
    └───────────────────┘         └───────────────────┘         └───────────────────┘
```

## Core Components

### Brain (Central Coordinator)

The Brain is the central orchestrator that:

1. **Parses Development Documents** - Reads YAML dev docs and creates a task DAG
2. **Manages Context** - Stores all project context (files, history, state) in Redis
3. **Schedules Tasks** - Assigns tasks to Workers based on dependencies and capabilities
4. **Routes AI Models** - Selects appropriate AI models based on task type and user preferences
5. **Checks Safety** - Validates commands before Workers execute them
6. **Manages Git** - Creates branches for tasks, handles merges

### Workers (Stateless Executors)

Workers are stateless task executors that:

1. **Receive Tasks** - Get task assignments from Brain via ZeroMQ
2. **Request Context** - Ask Brain for needed context (no local state)
3. **Execute Tasks** - Use Mini-Agent or other backends to complete tasks
4. **Report Results** - Send results and status updates back to Brain

### Communication

- **Protocol**: ZeroMQ with ROUTER/DEALER pattern
- **Transport**: IPC on Linux/macOS, TCP on Windows
- **Messages**: JSON-serialized Pydantic models

## Design Principles

### 1. Brain Holds All Truth

Workers never maintain state. All context, history, and coordination happens through Brain. This prevents:
- Workers hallucinating outdated context
- Conflicts from independent Worker actions
- Lost work due to Worker failures

### 2. User Controls AI Selection

Brain suggests AI models based on task type, cost, and performance, but users make final decisions considering:
- Budget constraints
- Network accessibility
- Quality requirements

### 3. Minimal Mini-Agent Modification

The Worker adapter wraps Mini-Agent with safety checks and context injection, requiring minimal changes to Mini-Agent itself.

### 4. Platform Abstraction

A platform adapter layer handles differences between:
- **Linux/macOS**: IPC transport, SIGTERM handling, /proc reading
- **Windows**: TCP transport, subprocess termination

## Data Flow

### Task Execution Flow

```
1. User submits development document
2. Brain parses document → Creates task DAG
3. Brain finds ready tasks (dependencies satisfied)
4. Brain selects Worker based on capabilities
5. Brain assigns task to Worker
6. Worker requests context from Brain
7. Worker executes task (via Mini-Agent)
8. Worker sends safety check requests → Brain approves/rejects
9. Worker completes task → Reports result
10. Brain updates context, marks task complete
11. Brain triggers dependent tasks → Repeat from step 3
```

### Context Flow

```
Worker                          Brain                           Redis
   │                              │                               │
   │──── Request Context ────────>│                               │
   │                              │──── Get Context ─────────────>│
   │                              │<─── Context Data ─────────────│
   │<─── Context Response ────────│                               │
   │                              │                               │
   │──── Update Context ─────────>│                               │
   │                              │──── Store Context ───────────>│
   │                              │                               │
```

## Scalability

### Current Phase (v0.1)

- Single-machine, multi-process
- Brain + multiple Workers on same host
- Shared filesystem for project files

### Future Phase

- Multi-machine distributed
- Workers on remote hosts connect via TCP
- Distributed file sync or shared storage
