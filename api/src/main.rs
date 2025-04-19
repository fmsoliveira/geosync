use axum::{routing::post, Router, Json, response::IntoResponse};
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::net::TcpListener;
use tokio::{io::{AsyncBufReadExt, BufReader}, process::Command as TokioCommand};

#[derive(Deserialize, Serialize)]
struct CrewRequest {
    address: String,
    first_date: String,
    second_date: String,
    current_year: String,
}

async fn run_crew(Json(payload): Json<CrewRequest>) -> impl IntoResponse {
    // Serializa o payload para JSON
    let json_args = serde_json::to_string(&payload).unwrap();

    // Usa TokioCommand para async streaming
    let mut child = TokioCommand::new("../geosync/.venv/bin/python")
        .arg("../geosync/src/geosync/main.py")
        .arg(json_args)
        .current_dir("../geosync")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Falha ao executar python");

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    

    let mut reader = BufReader::new(stdout).lines();
    let mut err_reader = BufReader::new(stderr).lines();

    let mut last_json = None;

    // Lê stdout do Python em tempo real
    while let Some(line) = reader.next_line().await.unwrap_or(None) {
        println!("[PYTHON STDOUT] {line}");
        // Se a linha for JSON válido, verifica se tem as duas keys
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&line) {
            if json_val.get("image_path_1").is_some() && json_val.get("image_path_2").is_some() {
                last_json = Some(json_val);
            }
        }
    }

    // Também imprime stderr em tempo real
    while let Some(line) = err_reader.next_line().await.unwrap_or(None) {
        eprintln!("[PYTHON STDERR] {line}");
    }

    // Espera o processo terminar
    let _ = child.wait().await;

    // Só responde ao cliente quando tiver o JSON do último agente
    match last_json {
        Some(json) => (axum::http::StatusCode::OK, Json(json)),
        None => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "No JSON output from crew"})),
        ),
    }
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/crew", post(run_crew));
    println!("Servidor Axum em [http://127.0.0.1](http://127.0.0.1):8080/crew");
    
    let listener = TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app.into_make_service()).await.unwrap();
}