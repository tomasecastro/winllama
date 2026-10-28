//! Subcomandos del CLI Winllama.

pub mod doctor;
pub mod list;
pub mod ps;
pub mod pull;
pub mod rm;
pub mod run;
pub mod serve;

/// Formatea un tamano en bytes a una cadena legible (KiB/MiB/GiB).
pub fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
