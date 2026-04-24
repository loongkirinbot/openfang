//! Gemini CLI backend driver.
//!
//! Spawns the `gemini` CLI (Google Gemini) as a subprocess with `--output-format stream-json`
//! and `--prompt`, parsing JSONL output into structured events.
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

/// LLM driver that delegates to the Gemini CLI.
pub struct GeminiCliDriver {
    cli_path: String,
    skip_permissions: bool,
    active_pids: Arc<DashMap<String, u32>>,
    message_timeout_secs: u64,
}

impl GeminiCliDriver {
    /// Create a new Gemini CLI driver.
    ///
    /// `cli_path` overrides the CLI binary path; defaults to `"gemini"` on PATH.
    /// `skip_permissions` adds `--approval-mode yolo` so the CLI runs non-interactively
    /// (required for daemon mode).
    pub fn new(cli_path: Option<String>, skip_permissions: bool) -> Self {
        if skip_permissions {
            warn!(
                "Gemini CLI driver: --approval-mode yolo enabled. \
                 The CLI will not prompt for approvals."
            );
        }

        Self {
            cli_path: cli_path
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "gemini".to_string()),
            skip_permissions,
            active_pids: Arc::new(DashMap::new()),
            message_timeout_secs: DEFAULT_MESSAGE_TIMEOUT_SECS,
        }
    }

    /// Create a new Gemini CLI driver with a custom timeout.
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

    /// Detect if the Gemini CLI is available on PATH.
    pub fn detect() -> Option<String> {
        let output = std::process::Command::new("gemini")
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

    /// Build Gemini CLI arguments.
    fn build_args(request: &CompletionRequest, prompt: &str, skip_permissions: bool) -> Vec<String> {
        let mut args = vec!["--output-format".to_string(), "stream-json".to_string()];

        // Model: strip provider prefix
        let model = request.model
            .strip_prefix("gemini-local/")
            .unwrap_or(&request.model);
        if !model.is_empty() && model != "auto" {
            args.push("--model".to_string());
            args.push(model.to_string());
        }

        // Permissions bypass (required for daemon/non-interactive mode)
        if skip_permissions {
            args.push("--approval-mode".to_string());
            args.push("yolo".to_string());
        }

        // No sandbox (use host environment)
        args.push("--sandbox=none".to_string());

        // Temperature via extra config
        if request.temperature > 0.0 {
            args.push("--config".to_string());
            args.push(format!("temperature={}", request.temperature));
        }

        // System prompt via env var
        // The Gemini CLI respects GEMINI_SYSTEM_PROMPT or we embed in prompt

        // Prompt as final positional argument
        args.push("--prompt".to_string());
        args.push(prompt.to_string());

        args
    }
}

// ============================================================================
// JSONL parsing
// ============================================================================

/// Parsed result from Gemini JSONL output.
#[allow(dead_code)]
struct GeminiParseResult {
    session_id: Option<String>,
    final_text: Option<String>,
    error_message: Option<String>,
    usage: TokenUsage,
    cost_usd: Option<f64>,
}

fn parse_gemini_jsonl(stdout: &str) -> GeminiParseResult {
    let mut session_id = None;
    let mut final_text: Option<String> = None;
    let mut error_message: Option<String> = None;
    let mut input_tokens = 0u64;
    let mut output_tokens = 0u64;
    let mut cost_usd = None;

    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };

        // Track session ID
        if let Some(sid) = event.get("session_id").or_else(|| event.get("sessionId")) {
            if let Some(s) = sid.as_str() {
                session_id.get_or_insert(s.to_string());
            }
        }

        let event_type = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        match event_type {
            "assistant" => {
                // Extract from nested message content
                if let Some(msg) = event.get("message") {
                    if let Some(content) = msg.get("content") {
                        if let Some(arr) = content.as_array() {
                            for block in arr {
                                let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                                if block_type == "text" || block_type == "content" {
                                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                                        let text = text.trim();
                                        if !text.is_empty() {
                                            final_text.get_or_insert_with(String::new).push_str(text);
                                            final_text.get_or_insert_with(String::new).push('\n');
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Also try flat content
                if let Some(content) = event.get("content").and_then(|v| v.as_str()) {
                    let content = content.trim();
                    if !content.is_empty() {
                        final_text.get_or_insert_with(String::new).push_str(content);
                    }
                }
            }
            "text" => {
                if let Some(part) = event.get("part") {
                    if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                        let text = text.trim();
                        if !text.is_empty() {
                            final_text.get_or_insert_with(String::new).push_str(text);
                        }
                    }
                } else if let Some(text) = event.get("content").and_then(|v| v.as_str()) {
                    let text = text.trim();
                    if !text.is_empty() {
                        final_text.get_or_insert_with(String::new).push_str(text);
                    }
                }
            }
            "result" => {
                if let Some(result) = event.get("result").and_then(|v| v.as_str()) {
                    let result = result.trim();
                    if !result.is_empty() {
                        final_text.get_or_insert_with(String::new).push_str(result);
                    }
                }
                // Collect usage
                if let Some(usage) = event.get("usage") {
                    if let Some(t) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                        input_tokens = t;
                    }
                    if let Some(t) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                        output_tokens = t;
                    }
                }
                if let Some(meta) = event.get("usage_metadata") {
                    if input_tokens == 0 {
                        if let Some(t) = meta.get("prompt_token_count").and_then(|v| v.as_u64()) {
                            input_tokens = t;
                        }
                    }
                    if output_tokens == 0 {
                        if let Some(t) = meta.get("candidates_token_count").and_then(|v| v.as_u64()) {
                            output_tokens = t;
                        }
                    }
                }
                // Error check
                let is_error = event
                    .get("is_error")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                    || event
                        .get("subtype")
                        .and_then(|v| v.as_str())
                        .map(|s| s.eq_ignore_ascii_case("error"))
                        .unwrap_or(false);
                if is_error {
                    if let Some(err) = event.get("error").and_then(|v| v.as_str()) {
                        error_message.get_or_insert_with(|| err.to_string());
                    } else if let Some(err) = event.get("message").and_then(|v| v.as_str()) {
                        error_message.get_or_insert_with(|| err.to_string());
                    }
                }
                // Cost
                if let Some(c) = event.get("total_cost_usd").or_else(|| event.get("cost_usd")) {
                    if let Some(f) = c.as_f64() {
                        cost_usd.get_or_insert(f);
                    }
                }
            }
            "step_finish" | "done" | "complete" => {
                if let Some(usage) = event.get("usage") {
                    if input_tokens == 0 {
                        if let Some(t) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                            input_tokens = t;
                        }
                    }
                    if output_tokens == 0 {
                        if let Some(t) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                            output_tokens = t;
                        }
                    }
                }
                if input_tokens == 0 || output_tokens == 0 {
                    if let Some(meta) = event.get("usage_metadata") {
                        if input_tokens == 0 {
                            if let Some(t) = meta.get("prompt_token_count").and_then(|v| v.as_u64()) {
                                input_tokens = t;
                            }
                        }
                        if output_tokens == 0 {
                            if let Some(t) = meta.get("candidates_token_count").and_then(|v| v.as_u64()) {
                                output_tokens = t;
                            }
                        }
                    }
                }
                if let Some(c) = event.get("total_cost_usd").or_else(|| event.get("cost_usd")) {
                    if let Some(f) = c.as_f64() {
                        cost_usd.get_or_insert(f);
                    }
                }
            }
            "error" => {
                let msg = event
                    .get("error")
                    .and_then(|v| v.as_str())
                    .or_else(|| event.get("message").and_then(|v| v.as_str()))
                    .or_else(|| event.get("content").and_then(|v| v.as_str()))
                    .unwrap_or_default();
                if !msg.is_empty() {
                    error_message.get_or_insert_with(|| msg.to_string());
                }
            }
            _ => {
                // For unknown event types, try content as text fallback
                if let Some(text) = event.get("content").and_then(|v| v.as_str()) {
                    let text = text.trim();
                    if !text.is_empty() {
                        final_text.get_or_insert_with(String::new).push_str(text);
                    }
                }
            }
        }
    }

    // Trim trailing newline from accumulated text
    if let Some(ref mut text) = final_text {
        *text = text.trim().to_string();
    }

    GeminiParseResult {
        session_id,
        final_text: final_text.filter(|t| !t.is_empty()),
        error_message,
        usage: TokenUsage {
            input_tokens,
            output_tokens,
        },
        cost_usd,
    }
}

/// Check if output suggests an auth problem.
fn is_gemini_auth_error(stderr: &str) -> bool {
    regex_lite::Regex::new(
        r"(?i)not\s+authenticated|please\s+authenticate|api[_ ]?key\s+(?:required|missing|invalid)|authentication\s+required|unauthorized|invalid\s+credentials|not\s+logged\s+in|login\s+required|run\s+`?gemini\s+auth",
    )
    .map(|re| re.is_match(stderr))
    .unwrap_or(false)
}

/// Check if output suggests an unknown session error.
fn is_gemini_unknown_session_error(stderr: &str) -> bool {
    regex_lite::Regex::new(
        r"(?i)unknown\s+session|session\s+.*\s+not\s+found|resume\s+.*\s+not\s+found|checkpoint\s+.*\s+not\s+found|cannot\s+resume|failed\s+to\s+resume",
    )
    .map(|re| re.is_match(stderr))
    .unwrap_or(false)
}

// ============================================================================
// LlmDriver impl
// ============================================================================

#[async_trait]
impl LlmDriver for GeminiCliDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let prompt = Self::build_prompt(&request);
        let args = Self::build_args(&request, &prompt, self.skip_permissions);

        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.args(&args);

        // System prompt via environment
        if let Some(ref sys) = request.system {
            cmd.env("GEMINI_SYSTEM_PROMPT", sys);
        }

        apply_env_filter(&mut cmd);

        if let Some(home) = home_dir() {
            cmd.env("HOME", &home);
        }
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        debug!(cli = %self.cli_path, "Spawning Gemini CLI");

        let mut child = cmd.spawn().map_err(|e| {
            LlmError::Http(format!(
                "Gemini CLI not found or failed to start ({}). \
                 Install: pip install google-gemini && gemini auth",
                e
            ))
        })?;

        let pid_label = request.model.clone();
        if let Some(pid) = child.id() {
            self.active_pids.insert(pid_label.clone(), pid);
            debug!(pid = pid, model = %pid_label, "Gemini CLI subprocess started");
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
                warn!(error = %e, model = %pid_label, "Gemini CLI subprocess failed");
                return Err(LlmError::Http(format!("Gemini CLI subprocess failed: {e}")));
            }
            Err(_elapsed) => {
                warn!(
                    timeout_secs = self.message_timeout_secs,
                    model = %pid_label,
                    "Gemini CLI subprocess timed out, killing process"
                );
                let _ = child.kill().await;
                return Err(LlmError::Http(format!(
                    "Gemini CLI subprocess timed out after {}s — process killed",
                    self.message_timeout_secs
                )));
            }
        };

        let stderr = String::from_utf8_lossy(&stderr_bytes).trim().to_string();

        if !status.success() {
            let code = status.code().unwrap_or(1);
            let stdout_str = String::from_utf8_lossy(&stdout_bytes).trim().to_string();
            let detail: &str = if !stderr.is_empty() { stderr.as_str() } else { &stdout_str };

            warn!(exit_code = code, model = %pid_label, stderr = %detail, "Gemini CLI exited with error");

            let message = if is_gemini_auth_error(detail) {
                format!("Gemini CLI is not authenticated. Run: gemini auth\nDetail: {detail}")
            } else if is_gemini_unknown_session_error(detail) {
                format!("Gemini session not found or expired: {detail}")
            } else {
                format!("Gemini CLI exited with code {code}: {detail}")
            };

            return Err(LlmError::Api { status: code as u16, message });
        }

        info!(model = %pid_label, "Gemini CLI subprocess completed successfully");

        let stdout = String::from_utf8_lossy(&stdout_bytes);
        let parsed = parse_gemini_jsonl(&stdout);

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
        let args = Self::build_args(&request, &prompt, self.skip_permissions);

        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.args(&args);

        if let Some(ref sys) = request.system {
            cmd.env("GEMINI_SYSTEM_PROMPT", sys);
        }

        apply_env_filter(&mut cmd);

        if let Some(home) = home_dir() {
            cmd.env("HOME", &home);
        }
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        debug!(cli = %self.cli_path, "Spawning Gemini CLI (streaming)");

        let mut child = cmd.spawn().map_err(|e| {
            LlmError::Http(format!(
                "Gemini CLI not found or failed to start ({}). \
                 Install: pip install google-gemini && gemini auth",
                e
            ))
        })?;

        let pid_label = format!("{}-stream", request.model);
        if let Some(pid) = child.id() {
            self.active_pids.insert(pid_label.clone(), pid);
            debug!(pid = pid, model = %pid_label, "Gemini CLI streaming subprocess started");
        }

        let stdout = child.stdout.take().ok_or_else(|| {
            self.active_pids.remove(&pid_label);
            LlmError::Http("No stdout from gemini CLI".to_string())
        })?;

        let reader = tokio::io::BufReader::new(stdout);
        let mut lines = reader.lines();

        let mut full_text = String::new();
        let mut final_usage = TokenUsage {
            input_tokens: 0,
            output_tokens: 0,
        };

        let timeout_duration = std::time::Duration::from_secs(self.message_timeout_secs);
        let stream_result = tokio::time::timeout(timeout_duration, async {
            while let Ok(Some(line)) = lines.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }

                let Ok(event) = serde_json::from_str::<serde_json::Value>(&line) else {
                    // Non-JSON — treat as raw text
                    full_text.push_str(&line);
                    let _ = tx.send(StreamEvent::TextDelta { text: line.clone() }).await;
                    continue;
                };

                let event_type = event
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                match event_type {
                    "assistant" => {
                        if let Some(msg) = event.get("message") {
                            if let Some(content) = msg.get("content") {
                                if let Some(arr) = content.as_array() {
                                    for block in arr {
                                        let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                                        if (block_type == "text" || block_type == "content")
                                            && block.get("text").is_some()
                                        {
                                            if let Some(chunk) = block.get("text").and_then(|v| v.as_str()) {
                                                if !chunk.is_empty() {
                                                    full_text.push_str(chunk);
                                                    let _ = tx.send(StreamEvent::TextDelta { text: chunk.to_string() }).await;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    "text" | "content" => {
                        let chunk = event.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        let nested = event.get("part").and_then(|p| p.get("text")).and_then(|v| v.as_str()).unwrap_or("");
                        let text_chunk = if !chunk.is_empty() { chunk } else { nested };
                        if !text_chunk.is_empty() {
                            full_text.push_str(text_chunk);
                            let _ = tx.send(StreamEvent::TextDelta { text: text_chunk.to_string() }).await;
                        }
                    }
                    "result" | "done" | "complete" => {
                        if let Some(usage) = event.get("usage") {
                            if let Some(t) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                                final_usage.input_tokens = t;
                            }
                            if let Some(t) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                                final_usage.output_tokens = t;
                            }
                        }
                        if let Some(meta) = event.get("usage_metadata") {
                            if final_usage.input_tokens == 0 {
                                if let Some(t) = meta.get("prompt_token_count").and_then(|v| v.as_u64()) {
                                    final_usage.input_tokens = t;
                                }
                            }
                            if final_usage.output_tokens == 0 {
                                if let Some(t) = meta.get("candidates_token_count").and_then(|v| v.as_u64()) {
                                    final_usage.output_tokens = t;
                                }
                            }
                        }
                        if let Some(text) = event.get("result").and_then(|v| v.as_str()) {
                            if full_text.is_empty() {
                                full_text = text.to_string();
                                let _ = tx.send(StreamEvent::TextDelta { text: text.to_string() }).await;
                            }
                        }
                    }
                    "error" => {
                        if let Some(err) = event.get("error").and_then(|v| v.as_str()) {
                            let _ = tx.send(StreamEvent::ToolExecutionResult {
                                id: String::new(),
                                name: "gemini".to_string(),
                                result_preview: err.to_string(),
                                is_error: true,
                            }).await;
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
                "Gemini CLI streaming subprocess timed out, killing process"
            );
            let _ = child.kill().await;
            return Err(LlmError::Http(format!(
                "Gemini CLI streaming subprocess timed out after {}s — process killed",
                self.message_timeout_secs
            )));
        }

        let status = child
            .wait()
            .await
            .map_err(|e| LlmError::Http(format!("Gemini CLI wait failed: {e}")))?;

        if !status.success() {
            let code = status.code().unwrap_or(1);
            return Err(LlmError::Api {
                status: code as u16,
                message: format!("Gemini CLI streaming exited with code {code}"),
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

/// Check if the Gemini CLI is available.
pub fn gemini_cli_available() -> bool {
    GeminiCliDriver::detect().is_some()
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
            model: "gemini-local/gemini-2.5-flash".to_string(),
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

        let prompt = GeminiCliDriver::build_prompt(&request);
        assert!(!prompt.contains("[System]"));
        assert!(!prompt.contains("You are helpful."));
        assert!(prompt.contains("[User]"));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_build_args_default() {
        let request = CompletionRequest {
            model: "auto".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            temperature: 0.0,
            system: None,
            thinking: None,
        };
        let args = GeminiCliDriver::build_args(&request, "hello", true);
        assert!(args.contains(&"--output-format".to_string()));
        assert!(args.contains(&"stream-json".to_string()));
        assert!(args.contains(&"--approval-mode".to_string()));
        assert!(args.contains(&"yolo".to_string()));
        assert!(args.contains(&"--prompt".to_string()));
        assert!(args.contains(&"hello".to_string()));
    }

    #[test]
    fn test_build_args_with_model() {
        let request = CompletionRequest {
            model: "gemini-local/gemini-2.5-pro".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            temperature: 0.0,
            system: None,
            thinking: None,
        };
        let args = GeminiCliDriver::build_args(&request, "hello", true);
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"gemini-2.5-pro".to_string()));
    }

    #[test]
    fn test_new_defaults_to_gemini() {
        let driver = GeminiCliDriver::new(None, true);
        assert_eq!(driver.cli_path, "gemini");
        assert_eq!(driver.message_timeout_secs, DEFAULT_MESSAGE_TIMEOUT_SECS);
        assert!(driver.active_pids().is_empty());
    }

    #[test]
    fn test_new_with_custom_path() {
        let driver = GeminiCliDriver::new(Some("/usr/local/bin/gemini".to_string()), true);
        assert_eq!(driver.cli_path, "/usr/local/bin/gemini");
    }

    #[test]
    fn test_new_with_empty_path() {
        let driver = GeminiCliDriver::new(Some(String::new()), true);
        assert_eq!(driver.cli_path, "gemini");
    }

    #[test]
    fn test_with_timeout() {
        let driver = GeminiCliDriver::with_timeout(None, true, 600);
        assert_eq!(driver.message_timeout_secs, 600);
        assert_eq!(driver.cli_path, "gemini");
    }

    #[test]
    fn test_parse_gemini_jsonl_basic() {
        let jsonl = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}
{"type":"result","result":"done","usage":{"input_tokens":10,"output_tokens":5}}
"#;
        let result = parse_gemini_jsonl(jsonl);
        assert!(result.final_text.is_some());
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
    }

    #[test]
    fn test_parse_gemini_jsonl_text_event() {
        let jsonl = r#"{"type":"text","content":"chunk1"}
{"type":"text","part":{"text":"chunk2"}}
{"type":"done","usage_metadata":{"prompt_token_count":10,"candidates_token_count":5}}
"#;
        let result = parse_gemini_jsonl(jsonl);
        assert_eq!(result.final_text, Some("chunk1chunk2".to_string()));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
    }

    #[test]
    fn test_parse_gemini_jsonl_error() {
        let jsonl = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"partial"}]}}
{"type":"result","is_error":true,"error":"something went wrong"}
"#;
        let result = parse_gemini_jsonl(jsonl);
        assert_eq!(result.error_message, Some("something went wrong".to_string()));
    }

    #[test]
    fn test_is_gemini_auth_error() {
        assert!(is_gemini_auth_error("please authenticate"));
        assert!(is_gemini_auth_error("not authenticated"));
        assert!(!is_gemini_auth_error("session expired"));
    }

    #[test]
    fn test_is_gemini_unknown_session_error() {
        assert!(is_gemini_unknown_session_error("unknown session"));
        assert!(is_gemini_unknown_session_error("session abc not found"));
        assert!(!is_gemini_unknown_session_error("not authenticated"));
    }
}
