# 📈 Multi-Agent Stock Analysis System with MLOps

## 🚀 Overview

This project is an **end-to-end stock analysis system** built using a **multi-agent architecture**, designed to generate **explainable BUY / SELL / HOLD decisions** rather than raw price predictions.

The system emphasizes **clean ML engineering, modular decision-making, experiment tracking, and production readiness**, instead of black-box deep learning models.

---

## 🎯 Problem Statement

Most stock analysis projects fall into one of two extremes:

- Simple rule-based scripts with no structure or scalability  
- Overhyped deep learning models that are difficult to explain and deploy  

The goal of this project was to build a **realistic, explainable, and production-oriented ML system** that:

- Uses **independent agents** for decision-making  
- Produces **transparent and auditable outputs**  
- Tracks system behavior using **MLOps tools**  
- Can be deployed reproducibly using **containers**

---

## 🛠️ Feature Engineering

Feature engineering is the process of converting **raw market data** into **meaningful, interpretable signals** that the system can reason about.

Instead of using raw stock prices directly, the project derives **statistical indicators** that capture **trend** and **risk**, which are essential for decision-making.

### 🔹 Simple Moving Averages (SMA 20 & SMA 50)

- **SMA 20** represents short-term market behavior  
- **SMA 50** represents medium-term market behavior  

Moving averages smooth out short-term noise and highlight underlying trends.

**How they are used in the system:**
- If price is above both SMAs → bullish trend  
- If price is below both SMAs → bearish trend  
- Otherwise → neutral or sideways market  

These features allow the system to reason about **trend direction** in a transparent and explainable way.

---

### 🔹 Rolling Volatility

Rolling volatility is computed as the **rolling standard deviation of returns** over a fixed window.

**Why volatility matters:**
- High volatility → higher uncertainty and risk  
- Low volatility → more stable market conditions  

Volatility is treated as a **risk feature**, independent of trend direction.  
This separation ensures the system understands that a market can be trending but still risky.

---

## 🤖 Multi-Agent System Design

The system follows a **custom multi-agent architecture**, implemented using **plain Python classes** (no external agent frameworks).

Each agent has:
- A **single responsibility**
- Clearly defined inputs and outputs
- Independent decision logic
- Explainable reasoning

---

### 🔹 Technical Analysis Agent

**Responsibility:** Identify the market trend.

**Inputs:**
- Close price  
- SMA 20  
- SMA 50  

**Outputs:**
- Trend signal: Bullish / Bearish / Neutral  
- Confidence score  
- Human-readable explanation  

This agent focuses purely on **trend detection** and does not consider risk.

---

### 🔹 Risk Analysis Agent

**Responsibility:** Assess market risk and uncertainty.

**Inputs:**
- Rolling volatility  

**Outputs:**
- Risk level: Low / Medium / High  
- Risk score  
- Explanation  

This agent is intentionally decoupled from trend logic, ensuring proper separation of concerns.

---

### 🔹 Decision Aggregation Agent

**Responsibility:** Combine outputs from multiple agents to make a final decision.

**Inputs:**
- Technical agent output  
- Risk agent output  

**Outputs:**
- Final decision: BUY / SELL / HOLD  
- Overall confidence  
- Reasoning  
- Agent-level summary  

This agent performs **multi-agent coordination**, resolving conflicts and balancing signals.

---
## 🔬 MLOps: Experiment Tracking with MLflow

MLflow is integrated to track **inference-time experiments**, not model training.

Instead of tracking loss or accuracy, the system logs **decision behavior** for each analysis run.

### What is tracked per run

- Input parameters (stock ticker, time period)
- Outputs from individual agents (trend signal, risk level)
- Metrics such as confidence score and risk score
- Artifacts containing the final decision in JSON format

### Why this matters

This allows:
- Auditing and explaining past decisions
- Comparing system behavior across different market conditions
- Debugging logic changes safely
- Applying MLOps principles beyond model training

This demonstrates a **production-oriented approach to ML systems**, where inference behavior is monitored just like training experiments.

---

## 🖥️ Interactive UI with Gradio

A **Gradio-based web interface** was developed to make the system interactive and user-friendly.

The UI allows users to:
- Select a stock ticker and time period
- View the final BUY / SELL / HOLD decision
- Inspect agent-level reasoning
- Visualize price trends along with moving averages

This confirms that the system works **end-to-end**, not just as backend logic.

---

## 🐳 Containerization with Docker

The entire application is containerized using **Docker** to ensure:

- Reproducibility across environments
- Stable dependency management
- Elimination of “works on my machine” issues

All dependencies are explicitly version-pinned to avoid runtime incompatibilities, which is critical for production systems.

---

## 🔧 Service Orchestration with Docker Compose

Docker Compose is used to orchestrate multiple services that together form the system:

- Stock analysis application (Gradio + ML pipeline)
- MLflow UI for experiment tracking
- Shared experiment storage (`mlruns`)

This setup mirrors real-world ML deployments where services are decoupled but interconnected.

Docker Compose enables:
- Single-command startup
- Clear service boundaries
- Consistent local deployment

---

## ⭐ Key Engineering Highlights

- Modular multi-agent architecture
- Explainable ML decision-making
- Feature engineering on time-series data
- Inference-level experiment tracking
- Dockerized deployment
- Multi-service orchestration using Docker Compose

---

## 🧰 Tech Stack

- **Language:** Python  
- **Data & ML:** Pandas, NumPy  
- **MLOps:** MLflow  
- **UI:** Gradio  
- **DevOps:** Docker, Docker Compose  

---

