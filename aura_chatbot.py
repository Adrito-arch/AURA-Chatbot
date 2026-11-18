import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# ----------------------------- Configuration -----------------------------
load_dotenv()

st.set_page_config(
    page_title="AURA — AI Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """You are a helpful, clear, and thoughtful AI assistant.
Answer accurately and naturally. Structure longer answers with headings and lists
when useful. If you are uncertain, say so rather than inventing information."""

# ----------------------------- Styling ------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #11131C;
        --slate: #1C2030;
        --surface: #292D40;
        --violet: #A78BFA;
        --coral: #FF8A65;
        --white: #F4F3F8;
        --muted: #A7A9B8;
        --line: rgba(244,243,248,.09);
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
          radial-gradient(ellipse at 75% 0%, rgba(167,139,250,.09), transparent 34%),
          var(--ink);
        color: var(--white);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1C2030 0%, #171A27 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    .brand {
        display:flex; align-items:center; gap:11px;
        padding: 7px 5px 22px;
    }
    .brand-mark {
        width:38px; height:38px; border-radius:13px;
        display:flex; align-items:center; justify-content:center;
        color:#171525; font-size:20px; font-weight:800;
        background:linear-gradient(135deg,#C4B5FD,#A78BFA 65%,#FF8A65);
        box-shadow:0 5px 22px rgba(167,139,250,.22);
    }
    .brand-name {
        font-family:'Manrope',sans-serif; font-size:19px;
        font-weight:800; letter-spacing:1.5px; color:var(--white);
    }
    .brand-sub { color:var(--muted); font-size:10px; letter-spacing:2px; margin-top:-2px; }
    .sidebar-label {
        color:#85899D; text-transform:uppercase; letter-spacing:1.5px;
        font-size:10px; font-weight:700; margin:15px 0 9px;
    }
    .hero { padding: 8vh 0 22px; text-align:center; }
    .hero-orb {
        width:62px;height:62px;border-radius:21px;margin:0 auto 23px;
        display:flex;align-items:center;justify-content:center;
        background:linear-gradient(145deg,rgba(167,139,250,.22),rgba(255,138,101,.12));
        border:1px solid rgba(167,139,250,.25);
        color:#C4B5FD;font-size:28px;
        box-shadow:0 12px 50px rgba(167,139,250,.10);
    }
    .hero h1 {
        font-family:'Manrope',sans-serif; font-size:clamp(30px,4vw,43px);
        line-height:1.15; font-weight:800; letter-spacing:-1.5px;
        color:var(--white); margin:0;
    }
    .hero h1 span {
        background:linear-gradient(90deg,#C4B5FD,#A78BFA 60%,#FFAD91);
        -webkit-background-clip:text;background-clip:text;color:transparent;
    }
    .hero p { color:var(--muted); font-size:14px; margin-top:13px; }
    .suggestion {
        background:linear-gradient(145deg,rgba(41,45,64,.88),rgba(28,32,48,.88));
        border:1px solid var(--line); border-radius:16px;
        padding:15px 16px; min-height:86px; margin-bottom:8px;
        transition:all .2s ease;
    }
    .suggestion:hover {
        border-color:rgba(167,139,250,.45);
        transform:translateY(-2px);
        box-shadow:0 10px 28px rgba(0,0,0,.15);
    }
    .suggestion b { color:var(--white); font-size:13px; display:block; margin-bottom:5px; }
    .suggestion span { color:var(--muted); font-size:11px; line-height:1.45; }
    [data-testid="stChatMessage"] {
        background:rgba(41,45,64,.55);
        border:1px solid var(--line);
        border-radius:18px; padding:16px 18px; margin-bottom:13px;
    }
    [data-testid="stChatInput"] {
        border:1px solid rgba(167,139,250,.24)!important;
        border-radius:18px!important;
        background:#1C2030!important;
        box-shadow:0 10px 35px rgba(0,0,0,.18);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color:rgba(167,139,250,.7)!important;
        box-shadow:0 0 0 3px rgba(167,139,250,.10);
    }
    .stButton > button {
        border-radius:12px; border:1px solid var(--line);
        background:rgba(41,45,64,.75); color:var(--white);
        transition:all .2s ease; font-weight:600;
    }
    .stButton > button:hover {
        border-color:rgba(167,139,250,.6); color:#D8CCFF;
        background:rgba(167,139,250,.10);
    }
    .stButton > button[kind="primary"] {
        background:linear-gradient(110deg,#A78BFA,#8B72D8);
        color:#171525;border:0;
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background:#11131C!important; color:var(--white)!important;
        border-color:var(--line)!important; border-radius:11px!important;
    }
    .caption-muted { color:#85899D; font-size:11px; text-align:center; margin-top:16px; }
    .status-pill {
        display:inline-flex; align-items:center; gap:7px;
        border:1px solid var(--line); border-radius:30px;
        padding:6px 11px; color:var(--muted); font-size:11px;
        background:rgba(41,45,64,.55);
    }
    .status-dot { width:6px;height:6px;border-radius:50%;background:#82D9A5; }
    hr { border-color:var(--line)!important; }
    #MainMenu, footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Session state ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    try:
        st.session_state.api_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        st.session_state.api_key = os.getenv("OPENROUTER_API_KEY", "")

if not st.session_state.api_key:
    st.error("OpenRouter API key not found. Check secrets.toml.")
    st.stop()       
if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL

# ----------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">✦</div>
          <div><div class="brand-name">AURA</div>
          <div class="brand-sub">YOUR AI WORKSPACE</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  New conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.text_input(
        "Model",
        value=st.session_state.model,
        key="model_input",
        help="Default: openrouter/free. You can enter another model slug supported by OpenRouter.",
    )
    st.session_state.model = st.session_state.model_input.strip() or DEFAULT_MODEL

    st.markdown('<div class="sidebar-label">Conversation</div>', unsafe_allow_html=True)
    st.caption(f"{len(st.session_state.messages)} messages in this chat")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-label">About</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#A7A9B8;font-size:12px;line-height:1.7'>"
        "AURA is a lightweight AI workspace powered by LangChain and OpenRouter."
        "<br><br><span style='color:#A78BFA'>●</span> Local interface"
        "<br><span style='color:#A78BFA'>●</span> Streaming responses"
        "<br><span style='color:#A78BFA'>●</span> Session-based chat history"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='caption-muted'>Your API key is not written into this Python file.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------- Main area ----------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-orb">✦</div>
          <h1>Ideas begin with<br><span>a conversation.</span></h1>
          <p>A calmer space to think, create, explore and get things done.</p>
          <div style="margin-top:18px">
            <span class="status-pill"><span class="status-dot"></span> AI workspace ready</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown(
            "<div class='suggestion'><b>✧ Explore an idea</b>"
            "<span>Break down a complex topic into clear, practical steps.</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='suggestion'><b>⌘ Build something</b>"
            "<span>Plan an app, product, workflow or creative project.</span></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            "<div class='suggestion'><b>↗ Solve a problem</b>"
            "<span>Think through challenges and compare possible approaches.</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='suggestion'><b>✎ Write with clarity</b>"
            "<span>Draft, refine and organize your thoughts.</span></div>",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        "<div style='padding:12px 0 22px'>"
        "<span class='status-pill'><span class='status-dot'></span> Conversation active</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# Render existing conversation
for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])

# ----------------------------- Chat pipeline -------------------------------
prompt = st.chat_input("Message AURA...")


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            llm = ChatOpenAI(
                model=st.session_state.model,
                openai_api_key=st.session_state.api_key,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=0.7,
                streaming=True,
                default_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "AURA AI Chatbot",
                },
            )

            history = [SystemMessage(content=SYSTEM_PROMPT)]

            for item in st.session_state.messages[:-1]:
                if item["role"] == "user":
                    history.append(HumanMessage(content=item["content"]))
                else:
                    history.append(AIMessage(content=item["content"]))

            history.append(HumanMessage(content=prompt))

            response_area = st.empty()
            full_response = ""

            for chunk in llm.stream(history):
                if chunk.content:
                    if isinstance(chunk.content, str):
                        full_response += chunk.content
                    elif isinstance(chunk.content, list):
                        full_response += "".join(
                            part.get("text", "")
                            for part in chunk.content
                            if isinstance(part, dict)
                        )

                    response_area.markdown(full_response + " ▍")

            response_area.markdown(
                full_response or "I couldn't generate a text response."
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as exc:
            st.error(
                "I couldn't reach the model. Check your API key, model name and internet connection."
            )
            with st.expander("Technical details"):
                st.code(str(exc))


