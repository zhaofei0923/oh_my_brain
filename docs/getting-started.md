# Getting Started

This guide will walk you through setting up OH MY BRAIN for your first multi-agent development session.

## Prerequisites

- Python 3.10 or higher
- Redis (optional, for persistent context storage)
- At least one AI provider API key (DeepSeek, Anthropic, OpenAI, etc.)

## Installation

### Using pip

```bash
pip install oh-my-brain
```

### Using uv (recommended)

```bash
uv pip install oh-my-brain
```

### From source

```bash
git clone https://github.com/your-org/oh-my-brain.git
cd oh-my-brain
pip install -e ".[dev]"
```

## Configuration

### 1. Set up API keys

Create a `.env` file in your project directory:

```bash
# Required: At least one AI provider
DEEPSEEK_API_KEY=your_deepseek_key_here

# Optional: Additional providers
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
```

### 2. Create a models configuration (optional)

```bash
cp config/models.example.yaml config/models.yaml
# Edit config/models.yaml to customize AI model selection
```

### 3. Create a Brain configuration (optional)

```bash
cp config/brain.example.yaml config/brain.yaml
# Edit config/brain.yaml for advanced settings
```

## Your First Project

### 1. Create a development document

OH MY BRAIN uses YAML development documents to define what needs to be built. Generate a template:

```bash
oh-my-brain doc template -o my_project.yaml
```

Edit `my_project.yaml` to describe your project modules and tasks.

### 2. Validate your document

```bash
oh-my-brain doc validate my_project.yaml
```

### 3. Start the Brain server

```bash
oh-my-brain brain start
```

### 4. Start Workers

In separate terminals, start one or more workers:

```bash
oh-my-brain worker start
```

### 5. Execute your development document

```bash
oh-my-brain doc run my_project.yaml
```

## Using Docker

For a quick setup with Docker Compose:

```bash
# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Start Brain + Workers
docker-compose up -d

# View logs
docker-compose logs -f
```

## Next Steps

- [Architecture](architecture.md) - Understand how Brain and Workers interact
- [Configuration](configuration.md) - Deep dive into configuration options
- [Development Document Format](dev-doc-format.md) - Learn how to write effective development documents
