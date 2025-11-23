

import time, json, re, os, random, functools
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

# If genai & MODEL_NAME already set in previous cells, use them; otherwise set defaults.
try:
    import google.generativeai as genai  # ensure installed earlier cell
except Exception as e:
    raise RuntimeError("Missing `google-generativeai`. Run your installation cell first.") from e

MODEL_NAME = globals().get("MODEL_NAME", "gemini-2.5-flash")
DEFAULT_MEM_DIR = "/kaggle/working/ccc_memory"
os.makedirs(DEFAULT_MEM_DIR, exist_ok=True)

# -------------------------
# Utilities: retry decorator
# -------------------------
def retry(max_attempts=3, backoff=0.6):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    sleep_time = backoff * (2 ** (attempt-1)) + random.random() * 0.1
                    time.sleep(sleep_time)
        return wrapper
    return deco

# -------------------------
# Data classes & Memory
# -------------------------
@dataclass
class PatientProfile:
    patient_id: str
    name: str
    diagnosis: str
    stage: str
    treatment_plan: List[str]
    medications: List[Dict[str, Any]]
    appointments: List[Dict[str, Any]]
    allergies: List[str]
    emergency_contacts: List[Dict[str, str]]

    def to_dict(self):
        return asdict(self)

@dataclass
class ConversationMemory:
    session_id: str
    messages: List[Dict[str, Any]]
    patient_state: Dict[str, Any]
    timestamp: str

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_recent_context(self, n: int = 8) -> str:
        recent = self.messages[-n:] if len(self.messages) > n else self.messages
        return "\n".join([f"{m['role']}: {m['content']}" for m in recent])

class MemoryManager:
    def __init__(self, mem_dir: str = DEFAULT_MEM_DIR):
        self.short_term: Dict[str, ConversationMemory] = {}
        self.long_term: Dict[str, PatientProfile] = {}
        self.mem_dir = mem_dir
        os.makedirs(self.mem_dir, exist_ok=True)

    def create_session(self, session_id: str, patient_id: str):
        memory = ConversationMemory(session_id=session_id, messages=[], patient_state={}, timestamp=datetime.now().isoformat())
        self.short_term[session_id] = memory
        return memory

    def get_session(self, session_id: str) -> Optional[ConversationMemory]:
        return self.short_term.get(session_id)

    def save_patient_profile(self, profile: PatientProfile):
        self.long_term[profile.patient_id] = profile
        path = os.path.join(self.mem_dir, f"patient_{profile.patient_id}.json")
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def get_patient_profile(self, patient_id: str) -> Optional[PatientProfile]:
        return self.long_term.get(patient_id)

    def persist_session(self, session: ConversationMemory):
        path = os.path.join(self.mem_dir, f"{session.session_id}.json")
        with open(path, "w") as f:
            json.dump({"session_id": session.session_id, "messages": session.messages, "patient_state": session.patient_state, "timestamp": session.timestamp}, f, indent=2)

    def summarize_and_prune(self, session: ConversationMemory, max_messages=60):
        if len(session.messages) <= max_messages:
            return
        # Summarize older half (keep recent)
        keep = max_messages // 2
        to_summarize = session.messages[:-keep]
        text = "\n".join([f'{m["role"]}: {m["content"]}' for m in to_summarize])
        prompt = f"Summarize the following conversation into 4 short bullet points (1-2 lines each):\n\n{text}"
        try:
            summary = genai.GenerativeModel(MODEL_NAME).generate_content(prompt).text
            session.messages = [{"role":"system_summary","content":summary,"timestamp":datetime.now().isoformat()}] + session.messages[-keep:]
        except Exception:
            # If summarization fails, fallback: truncate
            session.messages = session.messages[-max_messages:]

# -------------------------
# Tools (whitelisted actions)
# -------------------------
class ToolRegistry:
    def search_medications(self, medication_name: str) -> str:
        return f"✓ Found medication info for {medication_name} (summary placeholder)."

    def find_appointments(self, date_range: str) -> str:
        return "✓ Retrieved upcoming appointments (placeholder)."

    def log_symptom(self, symptom: str, severity: int) -> str:
        return f"✓ Logged symptom: {symptom} (severity: {severity}/10)."

    def connect_crisis_line(self) -> str:
        return "✓ Crisis resources provided."

# -------------------------
# BaseAgent with model call
# -------------------------
class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.model = genai.GenerativeModel(MODEL_NAME)

    @retry(max_attempts=3, backoff=0.4)
    def _call_model(self, prompt: str) -> str:
        time.sleep(0.25)
        resp = self.model.generate_content(prompt)
        return getattr(resp, "text", str(resp))

# -------------------------
# Specialized Agents
# -------------------------
class TreatmentNavigatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Treatment Navigator", "treatment coordination and medication management")

    def process(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        profile = context.get("patient_profile")
        meds = profile.medications[:2] if profile and profile.medications else []
        prompt = f"""You are Treatment Navigator. Keep response <=120 words. Provide empathetic, practical next steps and, if appropriate, output a JSON actions block after the reply using the marker ---ACTIONS---.

Patient diagnosis: {getattr(profile,'diagnosis','Unknown')}
Medications sample: {json.dumps(meds)}
User: {user_input}

Answer first in plain language, then possibly include:
---ACTIONS---
{{"actions":[{{"action":"log_symptom","params":{{"symptom":"nausea","severity":3}}}}]}} 
(If no action, omit the actions block.)
"""
        text = self._call_model(prompt)
        return {"agent": self.name, "response": text}

class EmotionalSupportAgent(BaseAgent):
    def __init__(self):
        super().__init__("Emotional Support Companion", "emotional wellness and mental health support")

    def process(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""You are Emotional Support Companion. Be warm, validating and concise (<=120 words). Detect crisis keywords and do NOT provide medical advice. If crisis present, include no actions and let orchestrator handle crisis.

User: {user_input}
"""
        text = self._call_model(prompt)
        crisis_keywords = ["want to die", "suicide", "end it all", "give up", "worthless"]
        crisis = any(k in user_input.lower() for k in crisis_keywords)
        return {"agent": self.name, "response": text, "crisis_detected": crisis}

class MedicalInformationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Medical Information Guide", "patient education and medical information")

    def process(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        profile = context.get("patient_profile")
        prompt = f"""You are Medical Information Guide. Explain medical concepts clearly and cite reputable sources. Keep under 150 words. NEVER diagnose. Context diagnosis: {getattr(profile,'diagnosis','Unknown')}

Question: {user_input}
"""
        text = self._call_model(prompt)
        return {"agent": self.name, "response": text}

# -------------------------
# Orchestrator: routing, action parsing, execution
# -------------------------
_ACTION_MARKER = r"---ACTIONS---\s*(\{.*\})\s*$"

class OrchestratorAgent:
    def __init__(self, memory_manager: MemoryManager, tools: ToolRegistry):
        self.memory_manager = memory_manager
        self.tools = tools
        self.agents = {
            "treatment": TreatmentNavigatorAgent(),
            "emotional": EmotionalSupportAgent(),
            "medical": MedicalInformationAgent()
        }
        self.model = genai.GenerativeModel(MODEL_NAME)

    def _intelligent_routing(self, user_input: str) -> List[str]:
        agents = []
        low = user_input.lower()
        if any(word in low for word in ["scared", "afraid", "worried", "anxious", "sad", "depressed", "feeling"]):
            agents.append("emotional")
        if any(word in low for word in ["what is", "explain", "how does", "why", "procedure", "treatment", "lumpectomy", "biopsy"]):
            agents.append("medical")
        if any(word in low for word in ["appointment", "medication", "when", "schedule", "nausea", "side effect", "chemo", "dose"]):
            agents.append("treatment")
        if not agents:
            # LLM fallback
            return self._llm_route(user_input)
        return agents

    def _llm_route(self, user_input: str, candidates=["treatment","emotional","medical"]) -> List[str]:
        prompt = f"You are a router. Choose up to two agents from {candidates} for this message. Return JSON like {{\"agents\":[\"emotional\",\"treatment\"]}}.\nUser message: '''{user_input}'''"
        try:
            raw = self.model.generate_content(prompt).text
            m = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
            if m:
                j = json.loads(m.group(1))
                return j.get("agents", []) or ["emotional"]
        except Exception:
            pass
        return ["emotional"]

    def _parse_actions_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            m = re.search(_ACTION_MARKER, text, flags=re.DOTALL)
            if m:
                payload = m.group(1)
                return json.loads(payload)
        except Exception:
            pass
        return None

    def _execute_actions(self, actions_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        if not actions_obj:
            return results
        for act in actions_obj.get("actions", []):
            name = act.get("action")
            params = act.get("params", {})
            if name == "log_symptom":
                r = self.tools.log_symptom(params.get("symptom",""), int(params.get("severity",0)))
            elif name == "search_medication":
                r = self.tools.search_medications(params.get("medication_name",""))
            elif name == "find_appointments":
                r = self.tools.find_appointments(params.get("date_range",""))
            elif name == "connect_crisis":
                r = self.tools.connect_crisis_line()
            else:
                r = f"❗ Unknown action '{name}' - not executed"
            results.append({"action": name, "result": r})
        return results

    def _synthesize(self, responses: Dict[str, Dict[str, Any]]) -> str:
        if len(responses) == 1:
            return list(responses.values())[0]["response"]
        combined = []
        for agent_key, response in responses.items():
            combined.append(response["response"])
        return "\n\n".join(combined)

    def _crisis_response(self) -> str:
        return ("🚨 I'm very concerned about what you've shared. Your safety is the most important thing right now.\n\n"
                "**IMMEDIATE HELP:**\n- 988 Suicide & Crisis Lifeline: Call or text 988\n- Crisis Text Line: Text HOME to 741741\n- Cancer Support Helpline: 1-800-227-2345\n\nPlease reach out right now. You don't have to face this alone. 💙")

    def route_query(self, user_input: str, session_id: str, patient_id: str) -> Dict[str, Any]:
        start = time.time()
        routing_agents = self._intelligent_routing(user_input)
        session = self.memory_manager.get_session(session_id)
        patient_profile = self.memory_manager.get_patient_profile(patient_id)
        context = {"patient_profile": patient_profile, "patient_state": session.patient_state if session else {}}
        responses = {}
        # Execute agents
        for key in routing_agents:
            agent = self.agents.get(key)
            if agent:
                try:
                    responses[key] = agent.process(user_input, context)
                except Exception as e:
                    responses[key] = {"agent": agent.name, "response": "Sorry, I'm having trouble responding right now."}
        # Check crisis
        crisis = any(r.get("crisis_detected", False) for r in responses.values())
        if crisis:
            final_response = self._crisis_response()
        else:
            # Parse and execute any actions from each agent
            tool_results = []
            for k,r in responses.items():
                actions = self._parse_actions_from_text(r.get("response",""))
                executed = self._execute_actions(actions) if actions else []
                if executed:
                    tool_results.extend(executed)
            final_response = self._synthesize(responses)
            if tool_results:
                final_response += "\n\nTools:\n" + "\n".join([f"- {t['action']}: {t['result']}" for t in tool_results])
        # Memory update & persistence
        if session:
            session.add_message("user", user_input)
            session.add_message("assistant", final_response)
            # summarise/prune occasionally
            self.memory_manager.summarize_and_prune(session, max_messages=80)
            self.memory_manager.persist_session(session)
        latency = time.time() - start
        return {"message": final_response, "agents_used": list(responses.keys()), "routing": {"reasoning":"hybrid"}, "crisis": crisis, "latency": latency}

# -------------------------
# Evaluation & System
# -------------------------
class EvaluationMetrics:
    def __init__(self):
        self.metrics = {"total_interactions":0, "crisis_detections":0, "agent_usage":{}, "response_times":[]}

    def log_interaction(self, interaction_data: Dict[str, Any]):
        self.metrics["total_interactions"] += 1
        for agent in interaction_data.get("agents_used", []):
            self.metrics["agent_usage"][agent] = self.metrics["agent_usage"].get(agent, 0) + 1
        if interaction_data.get("crisis"):
            self.metrics["crisis_detections"] += 1
        if "latency" in interaction_data:
            self.metrics["response_times"].append(interaction_data["latency"])

    def generate_report(self) -> str:
        rep = f"Total Interactions: {self.metrics['total_interactions']}\nCrisis Detections: {self.metrics['crisis_detections']}\nAgent Usage:\n"
        for a,c in self.metrics["agent_usage"].items():
            rep += f"- {a}: {c}\n"
        if self.metrics["response_times"]:
            rep += f"Avg response time: {sum(self.metrics['response_times'])/len(self.metrics['response_times']):.2f}s\n"
        return rep

class CancerCareCompanion:
    def __init__(self, mem_dir: str = DEFAULT_MEM_DIR):
        self.memory_manager = MemoryManager(mem_dir)
        self.tools = ToolRegistry()
        self.orchestrator = OrchestratorAgent(self.memory_manager, self.tools)
        self.evaluator = EvaluationMetrics()

    def create_session(self, patient_profile: PatientProfile):
        session_id = f"session_{patient_profile.patient_id}_{int(datetime.now().timestamp())}"
        self.memory_manager.save_patient_profile(patient_profile)
        self.memory_manager.create_session(session_id, patient_profile.patient_id)
        welcome = f"Hello {patient_profile.name}! 👋\nI'm your CancerCareCompanion. How are you feeling today?"
        # write initial system message
        session = self.memory_manager.get_session(session_id)
        session.add_message("system", welcome)
        self.memory_manager.persist_session(session)
        return {"session_id": session_id, "message": welcome}

    def send_message(self, session_id: str, patient_id: str, message: str):
        try:
            response = self.orchestrator.route_query(message, session_id, patient_id)
            self.evaluator.log_interaction(response)
            return response
        except Exception as e:
            return {"message": "Sorry — something went wrong. Please try again.", "agents_used":[], "routing":{}, "crisis": False}

    def get_report(self) -> str:
        return self.evaluator.generate_report()

# -------------------------
# Sample patient (you can replace with your object)
# -------------------------
sample_patient = PatientProfile(
    patient_id="P001",
    name="Sarah Johnson",
    diagnosis="Breast Cancer",
    stage="Stage II",
    treatment_plan=["Chemotherapy (AC-T)", "Lumpectomy", "Radiation"],
    medications=[{"name":"Ondansetron","dosage":"8mg","frequency":"PRN"}],
    appointments=[{"type":"Oncology Check-up","date":"2025-11-20","time":"10:00 AM","doctor":"Dr. Smith"}],
    allergies=["Penicillin"],
    emergency_contacts=[{"name":"John Johnson","relationship":"Spouse","phone":"555-0100"}]
)

# -------------------------
# Gradio UI
# -------------------------
import gradio as gr

system = CancerCareCompanion()
session_data = system.create_session(sample_patient)
session_id = session_data["session_id"]

def chat_with_agent(user_message, chat_history):
    if not user_message or not user_message.strip():
        return chat_history, ""
    resp = system.send_message(session_id, sample_patient.patient_id, user_message)
    badge = f"[{', '.join(resp.get('agents_used',[]))}]" if resp.get('agents_used') else ""
    chat_history = chat_history or []
    chat_history.append(("You", user_message))
    chat_history.append((f"Companion {badge}", resp["message"]))
    return chat_history, ""

def reset_session():
    global system, session_id
    system = CancerCareCompanion()
    new_session = system.create_session(sample_patient)
    session_id = new_session["session_id"]
    return [], ""

def export_report():
    return system.get_report()

with gr.Blocks() as demo:
    gr.Markdown("# 🎗️ CancerCareCompanion")
    with gr.Row():
        gr.Markdown("A supportive multi-agent assistant (Treatment • Emotional • Medical).")
    chatbot = gr.Chatbot(elem_id="ccc_chat", label="CancerCareCompanion")
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your message...")
        send = gr.Button("Send")
        reset_btn = gr.Button("Reset Session")
        export_btn = gr.Button("Export Report")
    send.click(chat_with_agent, [txt, chatbot], [chatbot, txt])
    txt.submit(chat_with_agent, [txt, chatbot], [chatbot, txt])
    reset_btn.click(reset_session, inputs=None, outputs=[chatbot, txt])
    export_btn.click(export_report, inputs=None, outputs=gr.Textbox(label="System Report"))

print("Launching Gradio interface...")
demo.launch() 
