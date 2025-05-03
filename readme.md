# 🌍 Geosync Project

**License: MIT**

## 🚀 Overview

This project is a **monorepo** that combines a high-performance Rust API with a Python [CrewAI](https://crewai.com/) application for geospatial intelligence.

- The **Rust API** (using [Axum](https://github.com/tokio-rs/axum)) serves as the backend interface, exposing endpoints to external clients.
- The API delegates geospatial processing tasks to a **CrewAI Python application**, which leverages advanced agents and tools for geospatial analysis.

## 🛰️ What does it do?

- Given a physical address, the system retrieves geospatial data for the corresponding location.
- The current implementation focuses on analyzing **vegetation change** by computing the NDVI (Normalized Difference Vegetation Index) between two dates, using satellite imagery.
- The system features an urban analysis tool that analyzes satellite images to automatically segment and identify buildings, using a fine-tuned SegFormer model (nvidia/segformer-b0-finetuned-ade-512-512).

## 🛠️ Technologies Used

- **🦀 Rust** (API backend)
  - [Axum](https://github.com/tokio-rs/axum) – Web framework
  - [Tokio](https://tokio.rs/) – Async runtime
  - [Serde](https://serde.rs/) – Serialization
- **🐍 Python** (CrewAI app)
  - [CrewAI](https://crewai.com/) – Multi-agent orchestration
  - [Geopandas](https://geopandas.org/) – Geospatial data processing
  - [Earth Engine](https://earthengine.google.com/) – Satellite imagery (via API)
  - [NumPy, Pillow, etc.] – Image processing
- **Other**
  - [🐳 Docker Compose](https://docs.docker.com/compose/) (optional, for orchestration)
  - 🛠️ Planned: [Vite](https://vitejs.dev/) + [React](https://react.dev/) for frontend

## ⚡ Installation

### Prerequisites

- 🦀 [Rust](https://www.rust-lang.org/tools/install)
- 🐍 [Python 3.10+](https://www.python.org/downloads/)
- 📦 [Poetry](https://python-poetry.org/) or `pip`/`venv`
- 🐳 (Optional) [Docker Compose](https://docs.docker.com/compose/)

### 1. Clone the repository

```bash
git clone [https://github.com/fmsoliveira/geosync.git](https://github.com/fmsoliveira/geosync.git)
cd geosync_project
```

### 2. Set up the Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Or, if using poetry:
# poetry install
```

### 3. Set up the Rust API

```bash
cd ../api
cargo build --release
```

### 4.(Optional) Set up environment variables

Create a .env file in geosync/ for API keys (e.g., Google Earth Engine, Geoapify, etc.)

### 5. Run the services

```bash
cd api
cargo run
```

The API will call the CrewAI Python app automatically as needed.

## 📡 Usage

Send a POST request to the API endpoint /crew with a JSON payload:

```json
{
  "address": "Avenida Doutor Alfredo Bensaúde, Lisboa, Portugal",
  "first_date": "2023-04-06",
  "second_date": "2024-04-13",
  "current_year": "2024"
}
```

### 6. Run the Frontend

```bash
cd frontend
npm install
npm run dev

The API will return the NDVI difference results and file paths to the generated satellite images.

## 🗺️ Roadmap

- [x] 🦀 Rust API integration with 🐍 Python CrewAI
- [x] 🌱 NDVI vegetation change detection
- [x] 🏙️ Urban analysis tool (🚧 🛠️ work in progress)
- [ ] 🖥️ Frontend application (Vite + React) for user interaction
- [ ] 🧑‍💻 Support for additional geospatial analyses
```
