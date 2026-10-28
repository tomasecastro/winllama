//! `winllama run <model>` — chat interactivo consumiendo `/api/chat`.

use std::io::{BufRead, Write};

use anyhow::Result;

use crate::client::{ChatMessage, Client};

/// Arranca un REPL de chat contra el daemon.
pub async fn execute(
    client: &Client,
    model: Option<String>,
    device: Option<String>,
    no_stream: bool,
) -> Result<()> {
    let model = match model {
        Some(m) => m,
        None => pick_default_model(client).await?,
    };

    println!("Winllama — chat con '{model}' (escribe /bye para salir)");
    if let Some(dev) = &device {
        println!("Dispositivo: {dev}");
    }
    if no_stream {
        println!("Modo: sin streaming (respuesta completa)");
    }
    println!();

    let stdin = std::io::stdin();
    let mut history: Vec<(String, String)> = Vec::new();

    loop {
        print!(">>> ");
        std::io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt == "/bye" || prompt == "/exit" {
            break;
        }

        let mut messages: Vec<ChatMessage> = Vec::with_capacity(history.len() + 1);
        for (role, content) in &history {
            messages.push(ChatMessage {
                role,
                content: content.as_str(),
            });
        }
        messages.push(ChatMessage {
            role: "user",
            content: prompt,
        });

        print!("winllama: ");
        std::io::stdout().flush()?;

        let mut answer = String::new();
        let result = if no_stream {
            match client.chat_once(&model, messages, device.as_deref()).await {
                Ok(text) => {
                    print!("{text}");
                    let _ = std::io::stdout().flush();
                    answer = text;
                    Ok(())
                }
                Err(err) => Err(err),
            }
        } else {
            client
                .chat_stream(&model, messages, device.as_deref(), |chunk| {
                    print!("{chunk}");
                    let _ = std::io::stdout().flush();
                    answer.push_str(chunk);
                })
                .await
        };

        println!();
        match result {
            Ok(()) => {
                if answer.is_empty() {
                    println!("(respuesta vacia del daemon)");
                } else {
                    history.push(("user".to_string(), prompt.to_string()));
                    history.push(("assistant".to_string(), answer));
                }
            }
            Err(err) => {
                eprintln!("error: {err:#}");
                println!("(se continua la sesion; verifica que el modelo este descargado)");
            }
        }
        println!();
    }

    println!("Fin de la sesion.");
    Ok(())
}

/// Pide `/api/tags` y elige un modelo si el usuario no paso uno.
async fn pick_default_model(client: &Client) -> Result<String> {
    let models = client
        .tags()
        .await
        .map_err(|e| anyhow::anyhow!("no se pudo obtener la lista de modelos: {e}"))?;
    match models.first() {
        Some(model) => {
            println!("Usando modelo por defecto: {}", model.name);
            Ok(model.name.clone())
        }
        None => anyhow::bail!(
            "no hay modelos locales; descarga uno con 'winllama pull <model>'"
        ),
    }
}
