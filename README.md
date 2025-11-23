 CancerCareCompanion — A Multi-Agent Support System for Cancer Patients

A compassionate AI assistant built using Google Gemini, Multi-Agent Architecture, Memory, Observability, and Tooling.
Designed to support cancer patients emotionally, medically, and logistically — all inside a single intelligent system.

 Overview

CancerCareCompanion is a multi-agent, AI-powered support assistant that helps patients manage:

✔ Treatment guidance

✔ Emotional well-being

✔ Medical education

✔ Symptom tracking

✔ Crisis detection

✔ Appointment & medication support

✔ Long-term + short-term memory

✔ Observability, evaluation, metrics

✔ Gradio UI for real-time interaction

This project was developed for the Google x Kaggle “Agents for Good” Challenge”, implementing all major concepts from the 5-Day AI Agents Intensive.

 Why CancerCareCompanion?

Cancer is overwhelming:
Patients often face confusion, fear, loneliness, and information overload.

Our goal is simple:

Provide a safe, supportive, trustworthy AI companion that helps patients understand, manage, and emotionally navigate their cancer journey.

The system combines 3 cooperative agents, memory, and evaluation tools to deliver personalized and responsible support.

🏗️ Architecture

The system follows a multi-agent orchestration design, with intelligent routing, tools, memory layers, observability, and a Gradio UI.

Below is the diagram you can use in GitHub:


🧩 Features Implemented
✔ 3 Multi-Agents (core requirement)

Treatment Navigator Agent
Medication details, appointments, side-effects, scheduling support

Emotional Support Agent
Validating, warm responses + crisis detection

Medical Information Agent
Clear explanations of medical terms & procedures (non-diagnostic)

✔ Agent Orchestration

Intelligent routing

LLM-fallback routing

Crisis override priority

Multi-agent synthesis

✔ Tools (Actions System)

Symptom logging

Medication lookup

Appointment retrieval

Crisis escalation

Agents output JSON actions which are parsed & executed.

✔ Memory (short-term + long-term)

Conversation history

Patient profile

Automatic summarization & pruning

Persistent storage in JSON

✔ Observability & Evaluation

The system logs:

Agent usage counts

Total interactions

Crisis detections

Latency metrics

Average response time

Exportable with Export Report button.

✔ Gradio User Interface

A clean, friendly interface:

Chat window

Reset session

Export system report

Real-time conversation updates


▶️ How to Run Locally
1. Install dependencies
pip install google-generativeai gradio

2. Add your Gemini API key
export GOOGLE_API_KEY="your-key-here"

3. Run the app
python app.py

Gradio will launch at:
http://127.0.0.1:7860

🧪 Evaluation Metrics (Built-In)

The system automatically generates:

Usage heatmaps

Agent selection patterns

Crisis detection logs

Average model latency

Export using Export Report button in the UI.

🌟 Acknowledgements

Google DeepMind — Gemini & ADK

Kaggle — Agents for Good challenge

5-Day AI Agents Intensive — Architecture inspiration


