//! Cliente HTTP hacia el daemon Winllama.
//!
//! Resuelve el puerto (explicito > config en `~/.winllama/config.json` > 11434)
//! y expone metodos tipados para cada endpoint del contrato (compatible Ollama).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Puerto por defecto del daemon (11434, por compatibilidad de contrato).
pub const DEFAULT_PORT: u16 = 11434;

/// Timeout para peticiones normales (listar, doctor, version).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Timeout para peticiones de generacion (pueden tardar en CPU).
const GENERATE_TIMEOUT: Duration = Duration::from_secs(600);

/// Ruta de la configuracion persistida del CLI.
pub fn config_path() -> PathBuf {
    let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join(".winllama").join("config.json")
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub port: Option<u16>,
}

/// Lee el puerto persistido en `~/.winllama/config.json`, si existe.
pub fn read_config_port() -> Option<u16> {
    let path = config_path();
    let raw = std::fs::read_to_string(&path).ok()?;
    let config: Config = serde_json::from_str(&raw).ok()?;
    config.port
}

/// Persiste el puerto activo en `~/.winllama/config.json`.
pub fn write_config_port(port: u16) -> Result<()> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }
    let config = Config { port: Some(port) };
    let raw = serde_json::to_string_pretty(&config)?;
    std::fs::write(&path, raw).with_context(|| format!("no se pudo escribir {}", path.display()))?;
    Ok(())
}

// ------------------------------------------------------------------ payloads

#[derive(Debug, Serialize)]
pub struct ChatRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<ChatMessage<'a>>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<&'a str>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    #[serde(default)]
    pub message: Option<ChunkMessage>,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkMessage {
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct TagsResponse {
    #[serde(default)]
    pub models: Vec<ModelTag>,
}

#[derive(Debug, Deserialize)]
pub struct ModelTag {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub modified_at: String,
    #[serde(default)]
    pub details: ModelDetails,
}

#[derive(Debug, Default, Deserialize)]
pub struct ModelDetails {
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

// -------------------------------------------------------------------- client

/// Cliente HTTP contra el daemon Winllama.
#[derive(Debug, Clone)]
pub struct Client {
    base_url: String,
    http: reqwest::Client,
}

impl Client {
    /// Crea un cliente resolviendo el puerto efectivo.
    pub fn new(host: &str, port: Option<u16>) -> Self {
        let resolved = port.or_else(read_config_port).unwrap_or(DEFAULT_PORT);
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("no se pudo construir el cliente HTTP");
        Self {
            base_url: format!("http://{host}:{resolved}"),
            http,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// `GET /api/version` — usado como handshake de disponibilidad.
    pub async fn version(&self) -> Result<String> {
        let url = format!("{}/api/version", self.base_url);
        let resp = self.http.get(&url).send().await.context("daemon no responde")?;
        let resp = resp.error_for_status().context("respuesta de error del daemon")?;
        let body: serde_json::Value = resp.json().await?;
        Ok(body
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string())
    }

    /// `GET /api/doctor` — diagnostico del daemon.
    pub async fn doctor(&self) -> Result<serde_json::Value> {
        let url = format!("{}/api/doctor", self.base_url);
        let resp = self.http.get(&url).send().await.context("daemon no responde")?;
        let resp = resp.error_for_status()?;
        Ok(resp.json().await?)
    }

    /// `GET /api/tags` — modelos locales.
    pub async fn tags(&self) -> Result<Vec<ModelTag>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.http.get(&url).send().await.context("daemon no responde")?;
        let resp = resp.error_for_status()?;
        let body: TagsResponse = resp.json().await?;
        Ok(body.models)
    }

    /// `POST /api/chat` (no streaming) — devuelve el texto completo.
    pub async fn chat(&self, model: &str, messages: Vec<ChatMessage<'_>>, device: Option<&str>) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);
        let payload = ChatRequest {
            model,
            messages,
            stream: false,
            device,
        };
        let resp = self
            .http
            .post(&url)
            .timeout(GENERATE_TIMEOUT)
            .json(&payload)
            .send()
            .await
            .context("daemon no responde")?;
        let status = resp.status();
        let body: serde_json::Value = resp.json().await?;
        if !status.is_success() {
            let msg = body
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("error del daemon");
            anyhow::bail!("{msg}");
        }
        Ok(body
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or_default()
            .to_string())
    }

    /// `POST /api/chat` con streaming NDJSON: entrega cada fragmento al callback.
    pub async fn chat_stream<F>(
        &self,
        model: &str,
        messages: Vec<ChatMessage<'_>>,
        device: Option<&str>,
        mut on_chunk: F,
    ) -> Result<()>
    where
        F: FnMut(&str),
    {
        use futures_util::StreamExt;

        let url = format!("{}/api/chat", self.base_url);
        let payload = ChatRequest {
            model,
            messages,
            stream: true,
            device,
        };
        let resp = self
            .http
            .post(&url)
            .timeout(GENERATE_TIMEOUT)
            .json(&payload)
            .send()
            .await
            .context("daemon no responde")?;
        let status = resp.status();
        if !status.is_success() {
            let body: serde_json::Value = resp.json().await.unwrap_or_default();
            let msg = body
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("error del daemon");
            anyhow::bail!("{msg}");
        }

        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut finished = false;
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = buffer.find('\n') {
                let line: String = buffer.drain(..=pos).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let parsed: ChatChunk = serde_json::from_str(line)
                    .with_context(|| format!("linea NDJSON invalida: {line}"))?;
                if let Some(err) = parsed.error {
                    anyhow::bail!("{err}");
                }
                if let Some(msg) = parsed.message {
                    if !msg.content.is_empty() {
                        on_chunk(&msg.content);
                    }
                }
                if parsed.done {
                    finished = true;
                    break;
                }
            }
            if finished {
                break;
            }
        }
        Ok(())
    }

    /// `POST /api/chat` sin streaming: devuelve el texto completo como `String`.
    pub async fn chat_once(
        &self,
        model: &str,
        messages: Vec<ChatMessage<'_>>,
        device: Option<&str>,
    ) -> Result<String> {
        self.chat(model, messages, device).await
    }
}
