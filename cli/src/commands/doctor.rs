//! `winllama doctor` — diagnostico del entorno.
//!
//! Consulta `/api/doctor` del daemon (dispositivos, motor, cache) y anade
//! chequeos locales del CLI (conectividad, version de Python disponible).

use anyhow::Result;

use crate::client::Client;

pub async fn execute(client: &Client) -> Result<()> {
    println!("== Winllama doctor ==");
    println!("CLI: {}", env!("CARGO_PKG_VERSION"));
    println!("Daemon: {}", client.base_url());

    // Chequeo 1: conectividad con el daemon.
    match client.version().await {
        Ok(version) => println!("[OK]   Daemon responde (version {version})"),
        Err(err) => {
            println!("[FAIL] Daemon no responde: {err}");
            println!("       Arrancalo con 'winllama serve'.");
            return Ok(());
        }
    }

    // Chequeo 2: diagnostico del daemon.
    let doctor = client.doctor().await?;

    let engine_ok = doctor
        .get("openvino_genai_available")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if engine_ok {
        println!("[OK]   Motor openvino-genai disponible en el daemon");
    } else {
        let err = doctor
            .get("openvino_genai_error")
            .and_then(|v| v.as_str())
            .unwrap_or("desconocido");
        println!("[FAIL] Motor openvino-genai no disponible: {err}");
        println!("       Instalalo con 'pip install openvino-genai' en el entorno del daemon.");
    }

    // Chequeo 3: dispositivos.
    if let Some(device) = doctor.get("device") {
        let selected = device.get("selected").and_then(|v| v.as_str()).unwrap_or("?");
        println!("[OK]   Dispositivo seleccionado: {selected}");
        if let Some(available) = device.get("available").and_then(|v| v.as_array()) {
            let names: Vec<String> = available
                .iter()
                .filter_map(|d| d.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect();
            println!("       Disponibles: {}", names.join(", "));
        }
        if device.get("degraded").and_then(|v| v.as_bool()).unwrap_or(false) {
            let warning = device
                .get("warning")
                .and_then(|v| v.as_str())
                .unwrap_or("degradacion sin detalle");
            println!("[WARN] {warning}");
        }
    }

    // Chequeo 4: modelos en cache.
    let models = client.tags().await?;
    if models.is_empty() {
        println!("[WARN] No hay modelos en el cache. Usa 'winllama pull <model>'.");
    } else {
        println!("[OK]   {} modelo(s) en el cache", models.len());
    }

    Ok(())
}
