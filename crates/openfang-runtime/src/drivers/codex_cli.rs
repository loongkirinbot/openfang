//! Codex CLI backend driver.
//!
//! Spawns the `codex` CLI (OpenAI Codex) as a subprocess in exec mode (`codex exec --json`),
//! delivering prompts via stdin and parsing JSONL output.
//!
//! Tracks active subprocess PIDs and enforces message timeouts to prevent
//! hung CLI processes from blocking agents indefinitely.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use crate::drivers::env_filter::apply_env_filter;
use async_trait::async_trait;
use dashmap::DashMap;
use openfang_types::message::{ContentBlock, Role, StopReason, TokenUsage};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt};
use tracing::{debug, info, warn};

/// Default subprocess timeout in seconds (5 minutes).
const DEFAULT_MESSAGE_TIMEOUT_SECS: u64 = 300;

/// Rollout noise pattern — filter out spurious stderr lines from Codex startup.
const CODEX_ROLLOUT_NOISE_RE: &str =
    r"^\d{4}-\d{2}-\d{2}T[^\s]+\s+ERROR\s+codex_core::rollout::list:\s+state db missing rollout path for thread\s+[a-z0-9-]+$";

/// LLM driver that delegates to the Codex CLI.
pub struct CodexCliDriver {
    cli_path: String,
    skip_permissions: bool,
    active_pids: Arc<DashMap<String, u32>>,
    message_timeout_secs: u64,
}

impl CodexCliDriver {
    /// Create a new Codex CLI driver.
    ///
    /// `cli_path` overrides the CLI binary path; defaults to `"codex"` on PATH.
    /// `skip_permissions` adds `--dangerously-bypass-approvals-and-sandbox` so the CLI
    /// runs non-interactively (required for daemon mode).
    pub fn new(cli_path: Option<String>, skip_permissions: bool) -> Self {
        if skip_permissions {
            warn!(
                "Codex CLI driver: --dangerously-bypass-approvals-and-sandbox enabled. \
                 The CLI will not prompt for tool approvals."
            );
        }

        Self {
            cli_path: cli_path
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "codex".to_string()),
            skip_permissions,
            active_pids: Arc::new(DashMap::new()),
            message_timeout_secs: DEFAULT_MESSAGE_TIMEOUT_SECS,
        }
    }

    /// Create a new Codex CLI driver with a custom timeout.
    pub fn with_timeout(
        cli_path: Option<String>,
        skip_permissions: bool,
        timeout_secs: u64,
    ) -> Self {
        let mut driver = Self::new(cli_path, skip_permissions);
        driver.message_timeout_secs = timeout_secs;
        driver
    }

    /// Get a snapshot of active subprocess PIDs.
    pub fn active_pids(&self) -> Vec<(String, u32)> {
        self.active_pids
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    /// Get the shared PID map for external monitoring.
    pub fn pid_map(&self) -> Arc<DashMap<String, u32>> {
        Arc::clone(&self.active_pids)
    }

    /// Detect if the Codex CLI is available on PATH.
    pub fn detect() -> Option<String> {
        let output = std::process::Command::new("codex")
            .arg("--version")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .ok()?;

        if output.status.success() {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            None
        }
    }

    /// Build a text prompt from the completion request messages.
    fn build_prompt(request: &CompletionRequest) -> String {
        let mut parts = Vec::new();

        for msg in &request.messages {
            let role_label = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };
            let text = msg.content.text_content();
            if !text.is_empty() {
                parts.push(format!("[{role_label}]\n{text}"));
            }
        }

        parts.join("\n\n")
    }

    /// Build the Codex exec command arguments.
    fn build_args(request: &CompletionRequest) -> Vec<String> {
        let mut args = vec!["exec".to_string(), "--json".to_string()];

        // Strip provider prefix if present (e.g. "codex-local/gpt-5.3-codex" -> "gpt-5.3-codex")
        let model = request.model
            .strip_prefix("codex-local/")
            .unwrap_or(&request.model);
        // Default model for codex-local is gpt-5.3-codex
        if !model.is_empty() && model != "gpt-5.3-codex" {
            args.push("--model".to_string());
            args.push(model.to_string());
        }

        // Temperature
        if request.temperature > 0.0 {
            args.push("-c".to_string());
            args.push(format!("temperature={}", request.temperature));
        }

        args.push("-".to_string()); // stdin prompt

        args
    }
}

// ============================================================================
// JSONL parsing
// ============================================================================

/// Parsed result from Codex JSONL output.
#[allow(dead_code)]
struct CodexParseResult {
    session_id: Option<String>,
    final_text: Option<String>,
    error_message: Option<String>,
    usage: TokenUsage,
}

fn parse_codex_jsonl(stdout: &str) -> CodexParseResult {
    let mut session_id = None;
    let mut final_text: Option<String> = None;
    let mut error_message: Option<String> = None;
    let mut input_tokens = 0u64;
    let mut output_tokens = 0u64;

    let re = regex_lite::Regex::new(CODEX_ROLLOUT_NOISE_RE).ok();

    for raw_line in stdout.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        // Filter rollout noise
        if let Some(ref regex) = re {
            if regex.is_match(line) {
                continue;
            }
        }

        let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };

        let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match event_type {
            "thread.started" => {
                if let Some(tid) = event.get("thread_id").and_then(|v| v.as_str()) {
                    session_id.get_or_insert(tid.to_string());
                }
            }
            "error" => {
                if let Some(err) = event.get("error") {
                    if let Some(msg) = err.get("message").and_then(|v| v.as_str()) {
                        error_message.get_or_insert_with(|| msg.to_string());
                    }
                }
            }
            "item.completed" => {
                if let Some(item) = event.get("item") {
                    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if item_type == "agent_message" {
                        if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                            final_text.get_or_insert_with(|| text.to_string());
                        }
                    }
                }
            }
            "turn.completed" => {
                if let Some(usage) = event.get("usage") {
                    if let Some(t) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                        input_tokens = t;
                    }
                    if let Some(t) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                        output_tokens = t;
                    }
                }
            }
            "turn.failed" => {
                if let Some(err) = event.get("error") {
                    if let Some(msg) = err.get("message").and_then(|v| v.as_str()) {
                        error_message.get_or_insert_with(|| msg.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    CodexParseResult {
        session_id,
        final_text,
        error_message,
        usage: TokenUsage {
            input_tokens,
            output_tokens,
        },
    }
}

/// Check if the error suggests an unknown/missing session (safe to ignore on retry).
fn is_codex_unknown_session_error(stderr: &str) -> bool {
    regex_lite::Regex::new(
        r"(?i)unknown (session|thread)|session .* not found|thread .* not found|conversation .* not found|missing rollout path|no rollout found for thread id",
    )
    .map(|re| re.is_match(stderr))
    .unwrap_or(false)
}

#[allow(dead_code)]
fn is_codex_transient_error(stderr: &str) -> bool {
    regex_lite::Regex::new(
        r"(?i)high demand|temporary errors|rate.limit|too many requests|429|server overloaded",
    )
    .map(|re| re.is_match(stderr))
    .unwrap_or(false)
}

// ============================================================================
// LlmDriver impl
// ============================================================================

#[async_trait]
impl LlmDriver for CodexCliDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let prompt = Self::build_prompt(&request);
        let args = Self::build_args(&request);

        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.args(&args);

        if self.skip_permissions {
            cmd.arg("--dangerously-bypass-approvals-and-sandbox");
        }

        // System prompt via env var (Codex respects SYSTEM_PROMPT)
        if let Some(ref sys) = request.system {
            cmd.env("SYSTEM_PROMPT", sys);
        }

        apply_env_filter(&mut cmd);

        if let Some(home) = home_dir() {
            cmd.env("HOME", &home);
        }
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        debug!(cli = %self.cli_path, skip_permissions = self.skip_permissions, "Spawning Codex CLI");

        // Write prompt to stdin
        let mut child = cmd.spawn().map_err(|e| {
            LlmError::Http(format!(
                "Codex CLI not found or failed to start ({}). \
                 Install: npm install -g @openai/codex && codex auth",
                e
            ))
        })?;

        let pid_label = request.model.clone();
        if let Some(pid) = child.id() {
            self.active_pids.insert(pid_label.clone(), pid);
            debug!(pid = pid, model = %pid_label, "Codex CLI subprocess started");
        }

        // Write prompt to stdin
        if let Some(ref mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            if let Err(e) = stdin.write_all(prompt.as_bytes()).await {
                warn!(error = %e, "Failed to write prompt to Codex stdin");
            }
            if let Err(e) = stdin.shutdown().await {
                warn!(error = %e, "Failed to shutdown Codex stdin");
            }
        }

        // Drain stdout and stderr concurrently while waiting for the process.
        let child_stdout = child.stdout.take();
        let child_stderr = child.stderr.take();

        let stdout_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(mut out) = child_stdout {
                let _ = out.read_to_end(&mut buf).await;
            }
            buf
        });
        let stderr_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(mut err) = child_stderr {
                let _ = err.read_to_end(&mut buf).await;
            }
            buf
        });

        let timeout_duration = std::time::Duration::from_secs(self.message_timeout_secs);
        let wait_result = tokio::time::timeout(timeout_duration, child.wait()).await;

        let stdout_bytes = stdout_task.await.unwrap_or_default();
        let stderr_bytes = stderr_task.await.unwrap_or_default();
        self.active_pids.remove(&pid_label);

        let status = match wait_result {
            Ok(Ok(status)) => status,
            Ok(Err(e)) => {
                warn!(error = %e, model = %pid_label, "Codex CLI subprocess failed");
                return Err(LlmError::Http(format!(
                    "Codex CLI subprocess failed: {e}"
                )));
            }
            Err(_elapsed) => {
                warn!(
                    timeout_secs = self.message_timeout_secs,
                    model = %pid_label,
                    "Codex CLI subprocess timed out, killing process"
                );
                let _ = child.kill().await;
                return Err(LlmError::Http(format!(
                    "Codex CLI subprocess timed out after {}s — process killed",
                    self.message_timeout_secs
                )));
            }
        };

        let stderr = String::from_utf8_lossy(&stderr_bytes).trim().to_string();

        if !status.success() {
            let code = status.code().unwrap_or(1);
            let stdout_str = String::from_utf8_lossy(&stdout_bytes).trim().to_string();
            let detail: &str = if !stderr.is_empty() { stderr.as_str() } else { &stdout_str };

            warn!(exit_code = code, model = %pid_label, stderr = %detail, "Codex CLI exited with error");

            let message = if detail.contains("not authenticated")
                || detail.contains("auth")
                || detail.contains("login")
                || detail.contains("credentials")
            {
                format!("Codex CLI is not authenticated. Run: codex auth\nDetail: {detail}")
            } else if is_codex_unknown_session_error(detail) {
                format!("Codex session not found or expired: {detail}")
            } else {
                format!("Codex CLI exited with code {code}: {detail}")
            };

            return Err(LlmError::Api { status: code as u16, message });
        }

        info!(model = %pid_label, "Codex CLI subprocess completed successfully");

        let stdout = String::from_utf8_lossy(&stdout_bytes);
        let parsed = parse_codex_jsonl(&stdout);

        if let Some(err) = parsed.error_message {
            return Err(LlmError::Api {
                status: 0,
                message: err,
            });
        }

        let text = parsed.final_text.unwrap_or_else(|| stdout.trim().to_string());

        Ok(CompletionResponse {
            content: vec![ContentBlock::Text {
                text,
                provider_metadata: None,
            }],
            stop_reason: StopReason::EndTurn,
            tool_calls: Vec::new(),
            usage: parsed.usage,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let prompt = Self::build_prompt(&request);
        let args = Self::build_args(&request);

        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.args(&args);

        if self.skip_permissions {
            cmd.arg("--dangerously-bypass-approvals-and-sandbox");
        }

        if let Some(ref sys) = request.system {
            cmd.env("SYSTEM_PROMPT", sys);
        }

        apply_env_filter(&mut cmd);

        if let Some(home) = home_dir() {
            cmd.env("HOME", &home);
        }
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        debug!(cli = %self.cli_path, "Spawning Codex CLI (streaming)");

        let mut child = cmd.spawn().map_err(|e| {
            LlmError::Http(format!(
                "Codex CLI not found or failed to start ({}). \
                 Install: npm install -g @openai/codex && codex auth",
                e
            ))
        })?;

        let pid_label = format!("{}-stream", request.model);
        if let Some(pid) = child.id() {
            self.active_pids.insert(pid_label.clone(), pid);
            debug!(pid = pid, model = %pid_label, "Codex CLI streaming subprocess started");
        }

        // Write prompt to stdin
        if let Some(ref mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(prompt.as_bytes()).await;
            let _ = stdin.shutdown().await;
        }

        let stdout = child.stdout.take().ok_or_else(|| {
            self.active_pids.remove(&pid_label);
            LlmError::Http("No stdout from codex CLI".to_string())
        })?;

        let reader = tokio::io::BufReader::new(stdout);
        let mut lines = reader.lines();

        let mut full_text = String::new();
        let mut final_usage = TokenUsage {
            input_tokens: 0,
            output_tokens: 0,
        };
        let re = regex_lite::Regex::new(CODEX_ROLLOUT_NOISE_RE).ok();

        let timeout_duration = std::time::Duration::from_secs(self.message_timeout_secs);
        let stream_result = tokio::time::timeout(timeout_duration, async {
            while let Ok(Some(line)) = lines.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }

                // Filter rollout noise
                if let Some(ref regex) = re {
                    if regex.is_match(&line) {
                        continue;
                    }
                }

                let Ok(event) = serde_json::from_str::<serde_json::Value>(&line) else {
                    // Non-JSON line — treat as raw text
                    full_text.push_str(&line);
                    let _ = tx.send(StreamEvent::TextDelta { text: line.clone() }).await;
                    continue;
                };

                let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

                match event_type {
                    "item.completed" => {
                        if let Some(item) = event.get("item") {
                            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if item_type == "agent_message" {
                                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                                    full_text.push_str(text);
                                    let _ = tx.send(StreamEvent::TextDelta { text: text.to_string() }).await;
                                }
                            }
                        }
                    }
                    "turn.completed" => {
                        if let Some(usage) = event.get("usage") {
                            if let Some(t) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                                final_usage.input_tokens = t;
                            }
                            if let Some(t) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                                final_usage.output_tokens = t;
                            }
                        }
                    }
                    "error" | "turn.failed" => {
                        if let Some(err) = event.get("error") {
                            if let Some(msg) = err.get("message").and_then(|v| v.as_str()) {
                                let _ = tx.send(StreamEvent::ToolExecutionResult {
                                    id: String::new(),
                                    name: "codex".to_string(),
                                    result_preview: msg.to_string(),
                                    is_error: true,
                                }).await;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }).await;

        self.active_pids.remove(&pid_label);

        if stream_result.is_err() {
            warn!(
                timeout_secs = self.message_timeout_secs,
                model = %pid_label,
                "Codex CLI streaming subprocess timed out, killing process"
            );
            let _ = child.kill().await;
            return Err(LlmError::Http(format!(
                "Codex CLI streaming subprocess timed out after {}s — process killed",
                self.message_timeout_secs
            )));
        }

        let status = child
            .wait()
            .await
            .map_err(|e| LlmError::Http(format!("Codex CLI wait failed: {e}")))?;

        if !status.success() {
            let code = status.code().unwrap_or(1);
            return Err(LlmError::Api {
                status: code as u16,
                message: format!("Codex CLI streaming exited with code {code}"),
            });
        }

        let _ = tx.send(StreamEvent::ContentComplete {
            stop_reason: StopReason::EndTurn,
            usage: final_usage,
        }).await;

        Ok(CompletionResponse {
            content: vec![ContentBlock::Text {
                text: full_text,
                provider_metadata: None,
            }],
            stop_reason: StopReason::EndTurn,
            tool_calls: Vec::new(),
            usage: final_usage,
        })
    }
}

/// Check if the Codex CLI is available.
pub fn codex_cli_available() -> bool {
    CodexCliDriver::detect().is_some()
}

fn home_dir() -> Option<std::path::PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(std::path::PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME").ok().map(std::path::PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_simple() {
        use openfang_types::message::{Message, MessageContent};

        let request = CompletionRequest {
            model: "codex-local/gpt-5.3-codex".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::text("Hello"),
            }],
            tools: vec![],
            max_tokens: 1024,
            temperature: 0.7,
            system: Some("You are helpful.".to_string()),
            thinking: None,
        };

        let prompt = CodexCliDriver::build_prompt(&request);
        assert!(!prompt.contains("[System]"));
        assert!(!prompt.contains("You are helpful."));
        assert!(prompt.contains("[User]"));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_build_args_default() {
        let request = CompletionRequest {
            model: "gpt-5.3-codex".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            temperature: 0.0,
            system: None,
            thinking: None,
        };
        let args = CodexCliDriver::build_args(&request);
        assert!(args.contains(&"--json".to_string()));
        assert!(args.contains(&"-".to_string()));
    }

    #[test]
    fn test_build_args_with_model() {
        let request = CompletionRequest {
            model: "codex-local/gpt-5.4".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            temperature: 0.0,
            system: None,
            thinking: None,
        };
        let args = CodexCliDriver::build_args(&request);
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"gpt-5.4".to_string()));
    }

    #[test]
    fn test_new_defaults_to_codex() {
        let driver = CodexCliDriver::new(None, true);
        assert_eq!(driver.cli_path, "codex");
        assert_eq!(driver.message_timeout_secs, DEFAULT_MESSAGE_TIMEOUT_SECS);
        assert!(driver.active_pids().is_empty());
    }

    #[test]
    fn test_new_with_custom_path() {
        let driver = CodexCliDriver::new(Some("/usr/local/bin/codex".to_string()), true);
        assert_eq!(driver.cli_path, "/usr/local/bin/codex");
    }

    #[test]
    fn test_new_with_empty_path() {
        let driver = CodexCliDriver::new(Some(String::new()), true);
        assert_eq!(driver.cli_path, "codex");
    }

    #[test]
    fn test_with_timeout() {
        let driver = CodexCliDriver::with_timeout(None, true, 600);
        assert_eq!(driver.message_timeout_secs, 600);
        assert_eq!(driver.cli_path, "codex");
    }

    #[test]
    fn test_pid_map_shared() {
        let driver = CodexCliDriver::new(None, true);
        let map = driver.pid_map();
        map.insert("test-agent".to_string(), 12345);
        assert_eq!(driver.active_pids().len(), 1);
        assert_eq!(driver.active_pids()[0], ("test-agent".to_string(), 12345));
    }

    #[test]
    fn test_parse_codex_jsonl_basic() {
        let jsonl = r#"{"type":"thread.started","thread_id":"abc123"}
{"type":"item.completed","item":{"type":"agent_message","text":"Hello world"}}
{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}
"#;
        let result = parse_codex_jsonl(jsonl);
        assert_eq!(result.session_id, Some("abc123".to_string()));
        assert_eq!(result.final_text, Some("Hello world".to_string()));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
    }

    #[test]
    fn test_parse_codex_jsonl_error() {
        let jsonl = r#"{"type":"thread.started","thread_id":"abc123"}
{"type":"error","error":{"message":"something went wrong"}}
"#;
        let result = parse_codex_jsonl(jsonl);
        assert_eq!(result.error_message, Some("something went wrong".to_string()));
    }

    #[test]
    fn test_is_codex_unknown_session_error() {
        assert!(is_codex_unknown_session_error("thread abc not found"));
        assert!(is_codex_unknown_session_error("Unknown session id"));
        assert!(!is_codex_unknown_session_error("some other error"));
    }

    #[test]
    fn test_is_codex_transient_error() {
        assert!(is_codex_transient_error("high demand"));
        assert!(is_codex_transient_error("rate limit exceeded"));
        assert!(!is_codex_transient_error("not authenticated"));
    }
}
