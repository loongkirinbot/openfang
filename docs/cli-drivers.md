# CLI-Based LLM Drivers

OpenFang ships with four LLM drivers that delegate to local CLI binaries instead of calling HTTP APIs. Each driver spawns the CLI as a subprocess, delivers prompts via stdin/arguments, and parses JSONL output back into structured responses.

These drivers require no API keys — authentication is handled by the CLI itself (OAuth, local credentials, etc.).

| Driver | Provider ID | CLI Binary | Auth Method |
|--------|-------------|------------|-------------|
| Claude Code | `claude-code` | `claude` | OAuth (`claude auth`) |
| Qwen Code | `qwen-code` | `qwen` | OAuth (`qwen auth`) |
| Codex CLI | `codex-local` | `codex` | OAuth (`codex auth`) |
| Gemini CLI | `gemini-local` | `gemini` | OAuth (`gemini auth`) |

---

## How CLI Drivers Work

All four drivers follow the same pattern:

```
Agent Message → OpenFang Runtime → CLI subprocess → JSONL stdout → CompletionResponse
```

1. **Prompt assembly**: Messages are joined into a plain-text prompt with role labels (`[User]`, `[Assistant]`, `[System]`)
2. **CLI invocation**: The driver spawns the CLI binary with appropriate flags
3. **Output parsing**: JSONL output lines are parsed into `CompletionResponse` or `StreamEvent`
4. **Security filtering**: API keys for other providers are stripped from the subprocess environment before spawning
5. **PID tracking**: Each subprocess PID is recorded so OpenFang can kill it on timeout
6. **Timeout enforcement**: Subprocesses that exceed the timeout are killed and an error is returned

---

## Installation

### Claude Code

```bash
npm install -g @anthropic-ai/claude-code
claude auth
```

### Qwen Code

```bash
npm install -g @qwen-code/qwen-code
qwen auth
```

### Codex CLI

```bash
npm install -g @openai/codex
codex auth
```

### Gemini CLI

```bash
pip install google-gemini
gemini auth
```

Verify the CLI is on your PATH:

```bash
claude --version    # Claude Code
qwen --version      # Qwen Code
codex --version     # Codex CLI
gemini --version    # Gemini CLI
```

---

## Authentication

All four CLIs use OAuth, so no API key is required. Run the auth command once:

| Driver | Auth Command |
|--------|-------------|
| Claude Code | `claude auth` |
| Qwen Code | `qwen auth` |
| Codex CLI | `codex auth` |
| Gemini CLI | `gemini auth` |

OpenFang automatically detects whether the CLI is authenticated. When the daemon runs as a service (without a login shell), the `HOME` environment variable is injected so the CLI can find its credentials at `~/.claude/`, `~/.qwen/`, `~/.codex/`, or `~/.gemini/`.

---

## Provider and Model IDs

### Claude Code — `claude-code`

**Provider ID:** `claude-code`

| Model ID | Description | Tier |
|----------|-------------|------|
| `claude-code/sonnet` | Claude Sonnet 4 (default) | Smart |
| `claude-code/opus` | Claude Opus 4 | Frontier |
| `claude-code/haiku` | Claude Haiku 4 | Fast |
| `claude-code/<custom>` | Any model the CLI accepts | — |

CLI invocation:
```
claude -p <prompt> --output-format json [--model <model>] [--system-prompt <sysprompt>] [--dangerously-skip-permissions]
```

---

### Qwen Code — `qwen-code`

**Provider ID:** `qwen-code`

| Model ID | Description | Tier |
|----------|-------------|------|
| `qwen-code/qwen3-coder` | Qwen 3 Coder (default) | Smart |
| `qwen-code/qwen-coder-plus` | Qwen Coder Plus | Frontier |
| `qwen-code/qwq-32b` | QWQ 32B reasoning | Fast |
| `qwen-code/<custom>` | Any model the CLI accepts | — |

CLI invocation:
```
qwen -p <prompt> --output-format json [--model <model>] [--yolo]
```

---

### Codex CLI — `codex-local`

**Provider ID:** `codex-local`

| Model ID | Description | Tier |
|----------|-------------|------|
| `gpt-5.3-codex` | GPT-5.3 Codex (default) | Smart |
| `gpt-5.4` | GPT-5.4 | Frontier |
| `gpt-5.3-codex-spark` | GPT-5.3 Codex Spark | Smart |
| `gpt-5` | GPT-5 | Smart |
| `o3` | OpenAI o3 | Frontier |
| `o4-mini` | OpenAI o4-mini | Fast |
| `gpt-5-mini` | GPT-5 mini | Balanced |
| `gpt-5-nano` | GPT-5 nano | Fast |
| `o3-mini` | OpenAI o3-mini | Fast |
| `codex-mini-latest` | Codex Mini | Fast |

CLI invocation:
```
codex exec --json [--model <model>] [-c temperature=N] [--dangerously-bypass-approvals-and-sandbox] -
```

Prompts are delivered via stdin (the `-` argument). System prompt is injected via the `SYSTEM_PROMPT` environment variable.

---

### Gemini CLI — `gemini-local`

**Provider ID:** `gemini-local`

| Model ID | Description | Tier |
|----------|-------------|------|
| `auto` | Auto-select best model (default) | Balanced |
| `gemini-2.5-pro` | Gemini 2.5 Pro | Frontier |
| `gemini-2.5-flash` | Gemini 2.5 Flash | Balanced |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | Fast |
| `gemini-2.0-flash` | Gemini 2.0 Flash | Balanced |
| `gemini-2.0-flash-lite` | Gemini 2.0 Flash Lite | Fast |

CLI invocation:
```
gemini --output-format stream-json [--model <model>] [--approval-mode yolo] [--sandbox=none] --prompt "<prompt>"
```

System prompt is injected via the `GEMINI_SYSTEM_PROMPT` environment variable.

---

## Configuration

### Agent Config

```toml
# ~/.openfang/config.toml

[[agents]]
name = "my-codex-agent"
model = { provider = "codex-local", model = "gpt-5.3-codex" }
system_prompt = "You are a helpful coding assistant."
```

### Driver Behavior Flags

When OpenFang spawns CLI drivers it passes two internal flags:

- **`skip_permissions = true`** (default for daemon mode): Adds the CLI flag to bypass interactive approval prompts. The CLI will not ask "Allow Read file X?" — OpenFang's own capability/RBAC system enforces access control instead.
  - Claude Code: `--dangerously-skip-permissions`
  - Codex CLI: `--dangerously-bypass-approvals-and-sandbox`
  - Gemini CLI: `--approval-mode yolo`
  - Qwen Code: `--yolo`

- **`base_url`**: Override the CLI binary path. Defaults to the CLI name on PATH.

```toml
# Use a custom codex binary path
[[agents]]
name = "dev-codex"
model = { provider = "codex-local", model = "gpt-5.3-codex", base_url = "/usr/local/bin/codex" }
```

### Subprocess Timeout

Each driver enforces a **5-minute timeout** by default. Subprocesses that exceed this are killed. The timeout can be configured per-driver via `with_timeout()` if needed.

---

## Environment Variables and Security

CLI drivers use `apply_env_filter()` to strip sensitive environment variables before spawning the subprocess. This prevents API keys for other LLM providers from leaking into the CLI subprocess environment.

### Variables Removed Unconditionally

```rust
OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY,
GROQ_API_KEY, DEEPSEEK_API_KEY, MISTRAL_API_KEY, TOGETHER_API_KEY,
FIREWORKS_API_KEY, OPENROUTER_API_KEY, PERPLEXITY_API_KEY, COHERE_API_KEY,
AI21_API_KEY, CEREBRAS_API_KEY, SAMBANOVA_API_KEY, HUGGINGFACE_API_KEY,
XAI_API_KEY, REPLICATE_API_TOKEN, BRAVE_API_KEY, TAVILY_API_KEY,
ELEVENLABS_API_KEY
```

### Variables Removed by Suffix

Any environment variable ending in `_SECRET`, `_TOKEN`, or `_PASSWORD` is also stripped — unless it has a driver-specific prefix:

| Driver | Preserved Prefix |
|--------|-----------------|
| Claude Code | `CLAUDE_` |
| Codex CLI | `CODEX_` |
| Gemini CLI | `GEMINI_` |
| Qwen Code | `QWEN_` |

The full environment (Node.js, NVM, SSL certificates, proxy settings) is kept intact — only known secrets are removed.

---

## Streaming

All four drivers support streaming. When the agent loop requests a streaming response, the CLI is invoked with a streaming output format:

| Driver | Streaming Format |
|--------|-----------------|
| Claude Code | `stream-json --verbose` |
| Qwen Code | `stream-json --verbose` |
| Codex CLI | (no special flag; JSONL contains `item.completed` events) |
| Gemini CLI | `stream-json` |

Each JSONL line is parsed and emitted as a `StreamEvent::TextDelta`. A final `StreamEvent::ContentComplete` signals the end of the stream.

---

## Error Handling

### Authentication Errors

Each driver detects auth failures and returns a descriptive error:

```
Claude Code CLI is not authenticated. Run: claude auth
Qwen Code CLI is not authenticated. Run: qwen auth
Codex CLI is not authenticated. Run: codex auth
Gemini CLI is not authenticated. Run: gemini auth
```

### Session Errors

Codex and Gemini CLI drivers detect session-not-found / session-expired errors (e.g., when resuming a session that no longer exists) and surface them clearly.

### Subprocess Timeouts

If a subprocess exceeds the timeout, it is killed and an error is returned:

```
<Driver> CLI subprocess timed out after 300s — process killed
```

### Non-JSON Output

If the CLI emits non-JSON output (e.g., warning messages on stderr that get mixed in), the driver falls back to treating the stdout as plain text.

---

## PID Tracking

Claude Code, Codex CLI, and Gemini CLI drivers track active subprocess PIDs in a `DashMap` keyed by agent/model label. This allows:

- External monitoring of running subprocesses
- Bulk termination of all subprocesses on shutdown
- Timeout enforcement per subprocess

Qwen Code does not use PID tracking (its `qwen -p` invocation is synchronous and returns immediately).

Access active PIDs via:

```rust
let driver: Arc<ClaudeCodeDriver> = ...;
for (label, pid) in driver.active_pids() {
    println!("running: {label} (PID {pid})");
}
```

---

## JSONL Output Reference

### Claude Code

```
{"type":"content","content":"Hello"}
{"type":"result","result":"Final answer","usage":{"input_tokens":10,"output_tokens":5}}
```

Newer CLI versions (≥2.x) emit nested content:
```
{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}
{"type":"result","result":"done","usage":{"input_tokens":10,"output_tokens":5}}
```

### Qwen Code

```
{"type":"content","content":"Hello"}
{"type":"result","result":"Final answer","usage":{"input_tokens":10,"output_tokens":5}}
```

### Codex CLI

```
{"type":"thread.started","thread_id":"abc123"}
{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}
{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}
{"type":"error","error":{"message":"..."}}
{"type":"turn.failed","error":{"message":"..."}}
```

Rollout noise (spurious stderr lines from Codex startup) is filtered by regex.

### Gemini CLI

```
{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}
{"type":"result","result":"done","usage":{"input_tokens":10,"output_tokens":5},"usage_metadata":{"prompt_token_count":10,"candidates_token_count":5}}
{"type":"error","error":"something went wrong"}
```

---

## Using with the OpenFang API

### Create an agent using a CLI driver

```bash
curl -X POST http://127.0.0.1:4200/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "codex-assistant",
    "model": {
      "provider": "codex-local",
      "model": "gpt-5.3-codex",
      "max_tokens": 4096,
      "temperature": 0.7
    },
    "system_prompt": "You are a senior software engineer.",
    "module": "builtin:chat"
  }'
```

### Send a message

```bash
curl -X POST http://127.0.0.1:4200/api/agents/<agent-id>/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a hello world in Rust."}'
```

### List available CLI providers

```bash
curl http://127.0.0.1:4200/api/models
# Filter for local providers:
curl http://127.127.0.1:4200/api/models?provider=codex-local
```

---

## Troubleshooting

### "CLI not found or failed to start"

The CLI binary is not on your PATH. Install it and ensure the directory containing the binary is in `$PATH`.

```bash
# Verify
which claude   # should return a path
codex --version  # should print version
```

### "Not authenticated"

Run the auth command for the respective CLI:

```bash
claude auth   # or qwen auth / codex auth / gemini auth
```

### Subprocess hangs / times out

The CLI is waiting for interactive input. Ensure `skip_permissions` is enabled (it is by default in daemon mode). Also verify `HOME` is accessible so the CLI can find its credentials.

### Responses are empty or truncated

The CLI may be writing to stderr instead of stdout. Check the OpenFang logs at trace level for the raw CLI output.

### High token counts (Claude Code)

Claude Code CLI v1.x uses `input_tokens` from `usage` in `turn.completed`. Claude Code CLI v2.x emits usage inside nested `message` structures. Both formats are handled.

### Codex: "rollout noise" in output

Codex CLI emits spurious `ERROR` lines to stderr during startup about missing rollout paths. These are filtered automatically by the driver.

### Permission prompt appears despite `skip_permissions`

The `--dangerously-skip-permissions` flag requires accepting the CLI permissions policy once manually first:

```bash
claude --dangerously-skip-permissions --print "test"
# Answer the prompts once to accept
```
