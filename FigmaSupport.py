import streamlit as st
import pandas as pd
import simplejson as json
import re
import base64
from autogen import UserProxyAgent, AssistantAgent

# ------------------------------
# Streamlit UI Setup
# ------------------------------
st.set_page_config(page_title="Figma QA Test Case Generator", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧪 AI QA Generator (Figma + Context)</h1>", unsafe_allow_html=True)

st.write("Upload Figma screenshots and provide context to generate UI-aware test cases.")

# ------------------------------
# User Inputs
# ------------------------------
user_story = st.text_area(
    "📜 Enter User Story:",
    "As a user, I want to log in so that I can access my account."
)

context_input = st.text_area(
    "🧠 Additional Context (Optional):",
    "Login screen includes email field, password field, login button, forgot password link."
)

uploaded_file = st.file_uploader(
    "🖼 Upload Figma Screenshot",
    type=["png", "jpg", "jpeg"]
)

# ------------------------------
# Convert Image to Base64
# ------------------------------
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

# ------------------------------
# LLM Config (Vision Enabled Model Required)
# ------------------------------
config_list = [
    {
        "model": "llava",  # Use vision-capable model (e.g., llava / gpt-4o / etc.)
        "api_type": "ollama",
        "stream": False,
    }
]

qa_agent = AssistantAgent(
    name="QA_Agent",
    max_consecutive_auto_reply=1,
    human_input_mode="NEVER",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    }
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config=False
)

# ------------------------------
# Generate Test Cases
# ------------------------------
def generate_test_cases(story, context, image_base64=None):

    image_instruction = ""
    if image_base64:
        image_instruction = f"""
        The following is a base64 encoded Figma screenshot:
        {image_base64}

        Analyze the screenshot carefully and:
        - Identify all UI components (buttons, fields, labels, dropdowns, error messages)
        - Understand layout relationships
        - Generate UI-specific test cases
        """

    prompt = f"""
    You are a Senior QA Engineer.

    USER STORY:
    {story}

    ADDITIONAL CONTEXT:
    {context}

    {image_instruction}

    Generate detailed UI-based test cases including:

    - Functional
    - UI/UX validation
    - Regression
    - Integration
    - Performance
    - Security
    - Positive
    - Negative
    - Edge cases

    IMPORTANT:
    - Base test cases on detected UI elements from screenshot.
    - Include field validations.
    - Include layout & visual validation checks.
    - Include navigation validations.

    Return ONLY valid JSON array.
    Do NOT include markdown.
    Do NOT include explanations.

    Format:
    [
      {{
        "ID": "TC001",
        "Summary": "",
        "Test Type": "",
        "Priority": "",
        "Component": "",
        "Step Description": "",
        "Expected Result": "",
        "Actual Result": "",
        "Pass/Fail Status": ""
      }}
    ]
    """

    response = user_proxy.initiate_chat(qa_agent, message=prompt)
    return extract_json_from_response(response)

# ------------------------------
# Extract JSON
# ------------------------------
def extract_json_from_response(response):
    chat_messages = response.chat_history

    for message in reversed(chat_messages):
        if 'content' in message and isinstance(message['content'], str):
            content = message['content'].strip()

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

            match = re.search(r'(\[\s*{.*?}\s*\])', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    return None
    return None

# ------------------------------
# Generate Button
# ------------------------------
if st.button("🚀 Generate Test Cases"):

    if uploaded_file:
        image_base64 = encode_image(uploaded_file)
    else:
        image_base64 = None

    with st.spinner("Analyzing screenshot and generating test cases..."):
        test_cases = generate_test_cases(user_story, context_input, image_base64)

        if test_cases:
            df = pd.DataFrame(test_cases)
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download as CSV",
                data=csv,
                file_name="figma_test_cases.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ Failed to generate test cases.")
