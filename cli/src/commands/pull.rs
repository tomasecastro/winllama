//! `winllama pull <model>` — descarga de modelos (Hito 4).
//!
//! El Hito 1 no implementa la descarga real desde Hugging Face: el endpoint
//! `/api/pull` del daemon todavia no existe. En vez de fingir exito, el comando
//! informa el estado y apunta al hito correspondiente.

use anyhow::Result;

use crate::client::Client;

pub async fn execute(client: &Client, model: String) -> Result<()> {
    println!("Descarga solicitada: {model}");
    println!("Daemon: {}", client.base_url());
    println!(
        "La descarga de modelos (Hito 4 / T-400) aun no esta implementada en el daemon.\n\
         Proximamente: 'winllama pull' descargara el IR desde Hugging Face via /api/pull."
    );
    Ok(())
}
