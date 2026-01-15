# Configuration

OH MY BRAIN uses YAML configuration files for Brain, Workers, and AI model settings.

## Configuration Files

| File | Purpose |
|------|---------|
| `config/brain.yaml` | Brain server settings |
| `config/models.yaml` | AI model pool and task mappings |
| `config/worker.yaml` | Worker settings |

## Brain Configuration

### Example `brain.yaml`

```yaml
# Server binding
host: "127.0.0.1"
port: 5555
transport: "auto"  # auto, ipc, tcp

# Redis for context storage
redis:
  enabled: true
  url: "redis://localhost:6379"

# Context management
context:
  max_tokens: 100000
  compression_threshold: 0.8

# Task scheduling
scheduler:
  strategy: "capability_match"
  task_timeout: 3600
  max_retries: 3

# Worker management
workers:
  heartbeat_interval: 10
  heartbeat_timeout: 30
  max_workers: 10

# Safety checks
safety:
  enabled: true
  dangerous_patterns:
    - 'rm\s+-rf\s+/'
    - 'sudo\s+'

# Git management
git:
  enabled: true
  base_branch: "develop"
  branch_prefix: "agent/"
```

### Key Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `transport` | Communication transport | `auto` |
| `context.max_tokens` | Maximum context size | 100000 |
| `scheduler.strategy` | Task assignment strategy | `capability_match` |
| `safety.enabled` | Enable command safety checks | `true` |

## AI Model Configuration

### Example `models.yaml`

```yaml
# Default model for unspecified tasks
default_model: "deepseek-chat"

# Model definitions
models:
  - id: "deepseek-chat"
    name: "DeepSeek Chat"
    provider: "deepseek"
    model_name: "deepseek-chat"
    api_key_env: "DEEPSEEK_API_KEY"
    base_url: "https://api.deepseek.com"
    cost_per_million_input: 0.14
    cost_per_million_output: 0.28
    capabilities:
      - "code"
      - "general"
    enabled: true

  - id: "claude-sonnet"
    name: "Claude 3.5 Sonnet"
    provider: "anthropic"
    model_name: "claude-3-5-sonnet-20241022"
    api_key_env: "ANTHROPIC_API_KEY"
    cost_per_million_input: 3.0
    cost_per_million_output: 15.0
    capabilities:
      - "code"
      - "architecture"
      - "review"
    enabled: true

# Task type to model mapping
task_mappings:
  feature:
    primary: "deepseek-chat"
    fallback: ["gpt-4o-mini"]

  architecture:
    primary: "claude-sonnet"
    fallback: ["claude-opus"]

  review:
    primary: "claude-sonnet"
    fallback: ["deepseek-chat"]

# Selection strategy
selection_strategy:
  priority: "cost"  # cost, quality, speed
  daily_budget: 10.0
```

### Supported Providers

| Provider | API Key Env | Base URL |
|----------|-------------|----------|
| DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| Anthropic | `ANTHROPIC_API_KEY` | Built-in |
| OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| MiniMax | `MINIMAX_API_KEY` | `https://api.minimax.chat/v1` |

## Worker Configuration

### Example `worker.yaml`

```yaml
worker_id: null  # Auto-generate
brain_address: "tcp://127.0.0.1:5555"

capabilities:
  - "python"
  - "javascript"
  - "docker"

max_concurrent_tasks: 1
heartbeat_interval_seconds: 10

mini_agent:
  path: null
  default_model: "deepseek-chat"
  non_interactive: true

resources:
  max_memory_mb: 4096
  task_timeout: 3600
```

## Environment Variables

All API keys should be set via environment variables:

```bash
# .env file
DEEPSEEK_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-xxx
REDIS_URL=redis://localhost:6379
```

## Command Line Overrides

Most settings can be overridden via command line:

```bash
# Override host and port
oh-my-brain brain start --host 0.0.0.0 --port 6666

# Override Brain address for Worker
oh-my-brain worker start --brain tcp://192.168.1.100:5555
```
