import gradio as gr
import joblib
import numpy as np
import requests
import traceback
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
from groq import Groq
import os

# ================================
# 📌 Load Model & Scaler from Files
# ================================
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# ================================
# 📌 DeepSeek API Setup
# ================================
DEEPSEEK_API_URL = "https://api.aimlapi.com/v1"



client1=Groq(api_key=os.getenv("GROQ_API_KEY_2"))
async def generate_explanation(credit_usage, age, debt_ratio, income, credit_lines, decision):
    try:
        prompt = f"""
        Given the following financial details:
        - Credit Usage: {credit_usage}
        - Age: {age}
        - Debt Ratio: {debt_ratio}
        - Monthly Income: {income}
        - Open Credit Lines: {credit_lines}

        Explain why the loan was {'APPROVED' if decision == 0 else 'REJECTED'}.
        """

        chat_completion = client1.chat.completions.create(
             model="deepseek-r1-distill-llama-70b",
             messages=[
                {"role": "system", "content": "You are a helpful assistant that explains financial decisions."},
                {"role": "user", "content": prompt}
            ],
            )

        # ✅ Extract response correctly
        response = chat_completion.choices[0].message.content
        return response if response else "No response from DeepSeek"

    except Exception as e:
        print(f"❌ DeepSeek API Error: {str(e)}")
        print(traceback.format_exc())
        return "An error occurred while generating the explanation."


# ================================
# 📌 Groq API Setup
# ================================
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

async def final_decision(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=50
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq API Error: {str(e)}")
        return "An error occurred while generating the explanation."


# ================================
# 📌 Function to Predict Loan Default
# ================================
async def predict_loan_default(credit_usage, age, debt_ratio, income, credit_lines):
    input_data = np.array([[credit_usage, age, debt_ratio, income, credit_lines]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    print("Prediction:", prediction)

    try:
        explanation = await generate_explanation(credit_usage, age, debt_ratio, income, credit_lines, prediction)
        hf_decision = await final_decision(explanation)
    except Exception as e:
        print(f"Error during API calls: {e}")
        explanation = "Error fetching explanation"
        hf_decision = "Error fetching final decision"

    return "APPROVED" if prediction == 0 else "REJECTED", explanation, hf_decision


# ================================
# 📌 Gradio UI Setup
# ================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏦 AI Loan Approval System
        🚀 **Smart, Transparent & Instant Loan Approvals**
        Enter applicant details to receive an **AI-powered loan decision** along with an explanation.
        """,
        elem_id="header",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Applicant Information")
            credit_usage = gr.Slider(0, 100, label="Credit Usage (%)")
            age = gr.Slider(18, 100, label="Age")
            debt_ratio = gr.Slider(0, 1, step=0.01, label="Debt Ratio")
            income = gr.Number(label="Monthly Income ($)")
            credit_lines = gr.Slider(0, 30, step=1, label="Open Credit Lines")

            submit_btn = gr.Button("📤 Submit Application", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Loan Decision & AI Explanation")
            with gr.Group():
                output_decision = gr.Textbox(label="✅ Decision", interactive=False)
                output_reasoning = gr.Textbox(label="💡 AI Explanation", interactive=False, lines=4)
                output_hf_decision = gr.Textbox(label="🤖 Final Assessment", interactive=False, lines=2)

    submit_btn.click(
        predict_loan_default,
        inputs=[credit_usage, age, debt_ratio, income, credit_lines],
        outputs=[output_decision, output_reasoning, output_hf_decision]
    )

    gr.Markdown(
        """
        ---
        **ℹ️ About This System:**
        - 📌 **AI-powered risk assessment** for loan applications
        - 🧠 **Real-time explanations** behind each decision
        - ⚡ **Quick & accurate processing** powered by ML models

        ✅ **Try it out now!**
        """,
        elem_id="footer",
    )

# Launch the app
demo.launch(debug=True)
