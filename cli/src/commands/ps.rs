//! `winllama ps` — modelos cargados en el daemon.
//!
//! El Hito 1 carga un solo modelo a la vez. El endpoint dedicado `/api/ps`
//! (el contrato lo usa para procesos en ejecucion) pertenece al Hito 2; por
//! ahora se consultan `/api/version`, `/api/doctor` y `/api/tags` para reportar
//! estado.

use anyhow::Result;

use crate::client::Client;

pub async fn execute(client: &Client) -> Result<()> {
    let version = client.version().await?;
    println!("Daemon en {} (version {version})", client.base_url());

    let doctor = client.doctor().await?;
    let selected = doctor
        .get("device")
        .and_then(|d| d.get("selected"))
        .and_then(|s| s.as_str())
        .unwrap_or("desconocido");
    let default_model = doctor
        .get("default_model")
        .and_then(|m| m.as_str())
        .unwrap_or("(ninguno)");

    println!("Dispositivo activo: {selected}");
    println!("Modelo por defecto: {default_model}");

    let models = client.tags().await?;
    if models.is_empty() {
        println!("No hay modelos cargados.");
    } else {
        println!("Modelos disponibles ({}):", models.len());
        for model in models {
            println!("  - {}", model.name);
        }
    }
    Ok(())
}
