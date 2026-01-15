# OH MY BRAIN

Multi-Agent Collaborative Development Framework

## Overview

OH MY BRAIN is a distributed multi-agent development framework that coordinates multiple AI-powered workers to collaboratively develop software projects. A central "Brain" orchestrates task distribution, context management, and code integration while "Workers" (powered by Mini-Agent or other backends) execute individual development tasks.

## Key Features

- **Centralized Coordination**: Brain manages all context, preventing Worker hallucinations
- **Parallel Execution**: Multiple Workers can work on different tasks simultaneously
- **User-Controlled AI Selection**: Configure which AI model to use for each task type
- **Cross-Platform**: Supports Linux, macOS, and Windows (via TCP)
- **Extensible**: Easy to add new AI providers and execution backends

## Quick Start

```bash
# Install
pip install oh-my-brain

# Generate a development document template
oh-my-brain doc template -o dev_doc.yaml

# Start the Brain server
oh-my-brain brain start

# In another terminal, start a Worker
oh-my-brain worker start
```

## Documentation

- [Getting Started](getting-started.md)
- [Architecture](architecture.md)
- [Configuration](configuration.md)
- [Development Document Format](dev-doc-format.md)
- [API Reference](api/index.md)
