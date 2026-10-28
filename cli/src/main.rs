//! Winllama CLI — cliente HTTP delgado del daemon Python (ADR-0004).
//!
//! Este binario **no enlaza OpenVINO**: solo habla HTTP con el daemon local en
//! `127.0.0.1:11434`. El motor de inferencia vive en `..\engine\` (Python).

mod client;
mod commands;

use clap::{Parser, Subcommand};

use client::Client;

/// CLI para LLMs locales en Windows con aceleracion Intel OpenVINO.
#[derive(Parser, Debug)]
#[command(name = "winllama", version, about, long_about = None)]
struct Cli {
    /// Puerto del daemon (default: leido de la config o 11434).
    #[arg(long, global = true)]
    port: Option<u16>,

    /// Host del daemon.
    #[arg(long, global = true, default_value = "127.0.0.1")]
    host: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Inicia un chat interactivo contra un modelo.
    Run {
        /// Nombre del modelo en el cache del daemon.
        model: Option<String>,
        /// Dispositivo de inferencia (CPU/GPU/NPU/AUTO).
        #[arg(long)]
        device: Option<String>,
        /// Desactiva el streaming (espera la respuesta completa).
        #[arg(long)]
        no_stream: bool,
    },
    /// Arranca el daemon de inferencia y hace handshake.
    Serve {
        /// Dispositivo de inferencia (CPU/GPU/NPU/AUTO).
        #[arg(long, default_value = "CPU")]
        device: String,
        /// Modelo por defecto.
        #[arg(long)]
        model: Option<String>,
        /// Ruta explicita al interprete de Python.
        #[arg(long)]
        python: Option<String>,
    },
    /// Descarga un modelo (delegado al daemon; placeholder del contrato).
    Pull { model: String },
    /// Lista los modelos locales.
    List,
    /// Elimina un modelo local.
    Rm { model: String },
    /// Muestra los modelos cargados en el daemon.
    Ps,
    /// Diagnostica el entorno (daemon + dispositivos + motor).
    Doctor,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let client = Client::new(&cli.host, cli.port);

    let result = match cli.command {
        Command::Run {
            model,
            device,
            no_stream,
        } => commands::run::execute(&client, model, device, no_stream).await,
        Command::Serve {
            device,
            model,
            python,
        } => commands::serve::execute(&client, device, model, python).await,
        Command::Pull { model } => commands::pull::execute(&client, model).await,
        Command::List => commands::list::execute(&client).await,
        Command::Rm { model } => commands::rm::execute(&client, model).await,
        Command::Ps => commands::ps::execute(&client).await,
        Command::Doctor => commands::doctor::execute(&client).await,
    };

    if let Err(err) = result {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
    Ok(())
}
