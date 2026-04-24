//! Shared environment filtering for CLI-based LLM drivers.
//!
//! Prevents API keys for other LLM providers from leaking into subprocess
//! environments. We keep the full environment (so Node.js, NVM, SSL, proxies
//! all work) and only remove known sensitive variables.

/// Environment variable names to remove unconditionally.
pub const SENSITIVE_ENV_EXACT: &[&str] = &[
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "OPENROUTER_API_KEY",
    "PERPLEXITY_API_KEY",
    "COHERE_API_KEY",
    "AI21_API_KEY",
    "CEREBRAS_API_KEY",
    "SAMBANOVA_API_KEY",
    "HUGGINGFACE_API_KEY",
    "XAI_API_KEY",
    "REPLICATE_API_TOKEN",
    "BRAVE_API_KEY",
    "TAVILY_API_KEY",
    "ELEVENLABS_API_KEY",
];

/// Suffixes that indicate a secret — remove any env var ending with these
/// unless it is prefixed with the driver-specific prefix (e.g. `CLAUDE_`).
pub const SENSITIVE_SUFFIXES: &[&str] = &["_SECRET", "_TOKEN", "_PASSWORD"];

/// Driver-specific env var prefixes to preserve even if they have sensitive suffixes.
pub const DRIVER_PREFIXES: &[&str] = &["CLAUDE_", "CODEX_", "GEMINI_"];

/// Apply security env filtering to a tokio Command.
///
/// Keeps the full environment intact and only removes known sensitive API keys
/// from other LLM providers. Uses `env_remove` to avoid breaking Node.js, NVM,
/// SSL, and proxy configuration.
pub fn apply_env_filter(cmd: &mut tokio::process::Command) {
    for key in SENSITIVE_ENV_EXACT {
        cmd.env_remove(key);
    }
    for (key, _) in std::env::vars() {
        // Preserve driver-specific prefixed vars
        for prefix in DRIVER_PREFIXES {
            if key.starts_with(prefix) {
                continue;
            }
        }
        let upper = key.to_uppercase();
        for suffix in SENSITIVE_SUFFIXES {
            if upper.ends_with(suffix) {
                cmd.env_remove(&key);
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitive_env_exact_contains_major_providers() {
        assert!(SENSITIVE_ENV_EXACT.contains(&"OPENAI_API_KEY"));
        assert!(SENSITIVE_ENV_EXACT.contains(&"ANTHROPIC_API_KEY"));
        assert!(SENSITIVE_ENV_EXACT.contains(&"GEMINI_API_KEY"));
        assert!(SENSITIVE_ENV_EXACT.contains(&"GROQ_API_KEY"));
        assert!(SENSITIVE_ENV_EXACT.contains(&"DEEPSEEK_API_KEY"));
    }

    #[test]
    fn driver_prefixes_includes_claude_codex_gemini() {
        assert!(DRIVER_PREFIXES.contains(&"CLAUDE_"));
        assert!(DRIVER_PREFIXES.contains(&"CODEX_"));
        assert!(DRIVER_PREFIXES.contains(&"GEMINI_"));
    }
}
