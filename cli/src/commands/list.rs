//! `winllama list` — lista los modelos locales via `/api/tags`.

use anyhow::Result;

use crate::client::Client;
use crate::commands::human_size;

/// Imprime una tabla con los modelos del cache del daemon.
pub async fn execute(client: &Client) -> Result<()> {
    let models = client.tags().await?;
    if models.is_empty() {
        println!("No hay modelos locales. Descarga uno con 'winllama pull <model>'.");
        return Ok(());
    }

    println!(
        "{:<28} {:>10} {:<12} {:<10} {:<10} {}",
        "NAME", "SIZE", "FORMAT", "PARAMS", "QUANT", "MODIFIED"
    );
    for model in models {
        println!(
            "{:<28} {:>10} {:<12} {:<10} {:<10} {}",
            model.name,
            human_size(model.size),
            model.details.format,
            model.details.parameter_size,
            model.details.quantization_level,
            model.modified_at
        );
    }
    Ok(())
}
