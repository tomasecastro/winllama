//! `winllama serve` — arranca el daemon Python (`python -m winllama_engine`)
//! y hace handshake contra `/api/version`.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::Command;

use crate::client::{write_config_port, Client};

/// Comprueba si ya hay un daemon respondiendo en el cliente dado.
pub async fn daemon_is_up(client: &Client) -> bool {
    client.version().await.is_ok()
}

/// Localiza el directorio `engine/` del proyecto (relativo al binario o al cwd).
fn find_engine_dir() -> Option<PathBuf> {
    // 1) Variable de entorno explicita.
    if let Ok(dir) = std::env::var("WINLLAMA_ENGINE_DIR") {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }
    // 2) Relativo al directorio de trabajo actual (repo root / cli).
    let candidates = [
        PathBuf::from("engine"),
        PathBuf::from("../engine"),
        PathBuf::from("src/Winllama/engine"),
        PathBuf::from("../Winllama/engine"),
    ];
    for candidate in candidates {
        if candidate.join("winllama_engine").is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// Arranca el daemon. Si ya esta corriendo, informa y termina.
pub async fn execute(
    client: &Client,
    device: String,
    model: Option<String>,
    python: Option<String>,
) -> Result<()> {
    if daemon_is_up(client).await {
        println!("El daemon ya esta en ejecucion en {}", client.base_url());
        return Ok(());
    }

    let python_exe = match python {
        Some(p) => p,
        None => {
            // T-227: si hay venv del engine, usalo ANTES del `python` global
            // (el global puede no tener openvino-genai; venv es el entorno
            // soportado del engine).
            if let Some(dir) = find_engine_dir() {
                let venv = dir.join(".venv").join("Scripts").join("python.exe");
                if venv.is_file() {
                    venv.to_string_lossy().to_string()
                } else {
                    "python".to_string()
                }
            } else {
                "python".to_string()
            }
        }
    };
    let engine_dir = find_engine_dir();

    let mut cmd = Command::new(&python_exe);
    cmd.arg("-m").arg("winllama_engine");
    cmd.arg("--device").arg(&device);
    if let Some(m) = &model {
        cmd.arg("--model").arg(m);
    }

    if let Some(dir) = &engine_dir {
        cmd.current_dir(dir);
        println!("Arrancando daemon desde {}", dir.display());
    } else {
        println!("Arrancando daemon (esperando 'winllama_engine' en el PYTHONPATH)");
    }

    // Heredar stdout/stderr del hijo para ver el log del daemon en consola.
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());
    cmd.kill_on_drop(false);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("no se pudo ejecutar '{python_exe} -m winllama_engine'"))?;
    println!("Daemon lanzado (pid {})", child.id().unwrap_or(0));

    // Handshake: esperar hasta ~20 s a que /api/version responda.
    let mut up = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if let Ok(version) = client.version().await {
            println!("Daemon listo en {} (version {version})", client.base_url());
            up = true;
            break;
        }
        if let Some(status) = child.try_wait()? {
            anyhow::bail!(
                "el daemon termino antes del handshake (codigo {:?}); revisa el log",
                status.code()
            );
        }
    }

    if !up {
        anyhow::bail!("el daemon no respondio al handshake en 20 s");
    }

    if let Some(port) = parse_port(client.base_url()) {
        write_config_port(port).context("no se pudo persistir el puerto del daemon")?;
    }

    println!("El daemon sigue en primer plano; pulsa Ctrl+C para detenerlo.");
    let status = child.wait().await?;
    println!("Daemon detenido ({status}).");
    Ok(())
}

/// Extrae el puerto de una URL `http://host:puerto`.
fn parse_port(base_url: &str) -> Option<u16> {
    base_url.rsplit(':').next()?.parse().ok()
}
