//! `winllama rm <model>` — elimina un modelo local (Hito 4).
//!
//! El endpoint `/api/delete` del daemon aun no existe; el comando verifica que
//! el modelo aparezca en `/api/tags` y avisa con honestidad que la eliminacion
//! se implementara en el Hito 4.

use anyhow::Result;

use crate::client::Client;

pub async fn execute(client: &Client, model: String) -> Result<()> {
    let models = client.tags().await?;
    if !models.iter().any(|m| m.name == model) {
        anyhow::bail!("el modelo '{model}' no esta en el cache local");
    }
    println!("Modelo '{model}' encontrado en el cache.");
    println!(
        "La eliminacion (Hito 4 / T-402, endpoint /api/delete) aun no esta implementada.\n\
         No se elimino ningun archivo."
    );
    Ok(())
}
