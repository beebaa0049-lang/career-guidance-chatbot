"""
CareerBuddy — AI Career Guidance Chatbot
Task 4 — EncoderX AI/ML Internship (Batch 02)

Tech Stack: Google Gemini API + Gradio
Features:
- Prompt engineering (persona, rules, guardrails)
- Conversation history management (sliding window)
- Error & rate-limit handling
- Interactive web UI with public deployment
"""

import gradio as gr
import google.generativeai as genai

# ============================================================
# 1. API KEY SETUP
#    Apni key yahan paste karein ya environment variable use karein
#    Free key: https://aistudio.google.com/apikey
# ============================================================
API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=API_KEY)

# ============================================================
# 2. PROMPT ENGINEERING
#    System prompt: persona + rules + output structure + guardrails
# ============================================================
SYSTEM_PROMPT = """
You are CareerBuddy, a professional Career Guidance Counselor.

RULES:
1. Only answer career, education, skills, jobs, resume and interview questions.
2. If the user asks something unrelated, politely redirect them to career topics.
3. Give a short summary first, then 3-5 actionable bullet points.
4. Be encouraging, concise and practical. Avoid generic advice.
5. End with a helpful follow-up question when relevant.
"""

# ============================================================
# 3. AI MODEL INTEGRATION
#    Gemini model with system instructions attached
# ============================================================
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT,
)

# ============================================================
# 4. BACKEND — QUERY PROCESSING + HISTORY + RESPONSE GENERATION
# ============================================================
def predict(message, history):
    """
    Core chatbot pipeline:
    user input -> conversation history formatting ->
    Gemini API call -> response / graceful error handling
    """
    if not message.strip():
        return "Please type a career-related question 🙂"

    # Conversation history management (last 10 messages — sliding window)
    chat_history = []
    for item in history[-10:]:
        if isinstance(item, dict):  # new Gradio messages format
            role = "user" if item["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [item["content"]]})
        else:  # old Gradio tuple format
            chat_history.append({"role": "user", "parts": [item[0]]})
            if item[1]:
                chat_history.append({"role": "model", "parts": [item[1]]})

    try:
        session = model.start_chat(history=chat_history)
        response = session.send_message(message)
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "⏳ Rate limit reached — please wait 30 seconds and try again."
        return f"⚠️ Error: {str(e)[:200]}"

# ============================================================
# 5. FRONTEND + DEPLOYMENT
#    Gradio chat interface with public share link
# ============================================================
demo = gr.ChatInterface(
    fn=predict,
    title="🎓 CareerBuddy — AI Career Guidance Assistant",
    description="Ask me anything about careers, skills, resumes or interviews!",
    examples=[
        ["What skills do I need for data science?"],
        ["How do I prepare for my first job interview?"],
        ["Give me a roadmap to learn machine learning"],
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True generates a public link
