# Anthropic API Proxy for OpenAI(Compatible) Models 🔄

**Use Anthropic clients (like Claude Code) with OpenAI(Compatible) models, or direct Anthropic backends.** 🤝

A proxy server that lets you use Anthropic clients with OpenAI(Compatible) models, or Anthropic models themselves (a transparent proxy of sorts), all via LiteLLM. 🌉


![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- OpenAI(or Compatible) API key 🔑
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

#### From source

1. **Clone this repository**:
   ```bash
   git clone -b baseurl https://github.com/byebyebruce/claude-code-proxy.git 
   cd claude-code-proxy
   ```

2. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Run the server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```
   *(`--reload` is optional, for development)*

#### From docker

```bash
docker-compose up -d
```

### Usage Examples 🎮

1. **Configure Claude Code**:

configure the claude code to use the proxy

```bash
# ~/.bashrc or ~/.zshrc
claude.gpt52() {
  export ANTHROPIC_BASE_URL="http:127.0.0.1:8082/https://api.openai.com/v1"
  export ANTHROPIC_MODEL="gpt-5.2"
  export ANTHROPIC_AUTH_TOKEN="sk-"
  export ANTHROPIC_DEFAULT_OPUS_MODEL="$ANTHROPIC_MODEL"
  export ANTHROPIC_DEFAULT_SONNET_MODEL="$ANTHROPIC_MODEL"
  export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ANTHROPIC_MODEL"
  export CLAUDE_CODE_SUBAGENT_MODEL=$ANTHROPIC_MODEL
  command claude "$@"
}
```

reload the shell

```bash
source ~/.bashrc # or ~/.zshrc
```

2. **Start the claude code**:
```bash
claude.gpt52
```