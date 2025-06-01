import streamlit as st
from streamlit_chat import message
from streamlit_feedback import streamlit_feedback
from dotenv import load_dotenv
import os
from loguru import logger
from src.utils.config import load_config
from src.utils.logging_config import setup_logging
from src.agents.conversation import ConversationManager

from ibm_watsonx_ai import APIClient

from ibm_watsonx_ai.foundation_models.moderations import Guardian

detectors = {
  "granite_guardian": {"threshold": 0.4},
  "hap": {"threshold": 0.4},
  "pii": {},
}

credentials = {
                    "url": os.getenv("ibm_url"),
                    "apikey": os.getenv("ibm_api_key")
                }
api_client = APIClient(credentials)
api_client.set.default_project(os.getenv("ibm_project_id"))

guardian = Guardian(
  detectors=detectors,  # required,
  api_client=api_client
)


# Load environment variables and setup
load_dotenv()
setup_logging()
logger.info("Starting CalPERS Multi-Agent Assistant")

# Try to setup tracing, but continue if it fails
try:
    from src.utils.tracing_config import setup_tracing
    setup_tracing()
    logger.info("Tracing initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize tracing: {str(e)}. Application will continue without tracing.")

# Load configuration and initialize conversation manager
config = load_config()
conversation_manager = ConversationManager()

def initialize_session_state():
    """Initialize session state variables"""
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Greetings! I am your CalPERS Assistant. How can I help you today?"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Welcome to the CalPERS Assistant Chat!"]
    if 'classifications' not in st.session_state:
        st.session_state['classifications'] = []

def setup_page():
    """Setup the Streamlit page configuration"""
    st.set_page_config(
        page_title="CalPERS Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Load custom CSS
    with open('style/final.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Add custom CSS to center-align main content
    st.markdown("""
        <style>
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 100%;
            }
            .stApp {
                max-width: 100%;
            }
            /* Center align main content */
            .main > div {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 2rem;
            }
            /* Center align the chat messages */
            .stChatMessage {
                max-width: 1000px;
                margin: 0 auto;
            }
            /* Center align the input form */
            .stForm {
                max-width: 1000px;
                margin: 0 auto;
            }
            /* Center align the radio buttons */
            .stRadio > div {
                max-width: 1000px;
                margin: 0 auto;
            }
        </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the application header with logo and title"""
    # Center the logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image('image/image.png', width=200)
    
    # Display title and subtitle
    st.markdown("<p style='text-align: center; color: black; font-size:22px;'><span style='font-weight: bold'></span>GenAI-Driven Rule Intelligence: Transform Policy Logic into Actionable Design with Agentic Precision</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: blue;margin-top: -10px ;font-size:18px;'><span style='font-weight: bold'></span>Boost Rule Clarity, Cut Manual Review Time, and Optimize Legacy Systems with Purpose-Built GenAI Agents</p>", unsafe_allow_html=True)
    st.markdown("<hr style=height:2.5px;margin-top:0px;width:100%;background-color:gray;>", unsafe_allow_html=True)

def setup_sidebar():
    """Setup the sidebar with model information and controls"""
    with st.sidebar:
        st.markdown("<p style='text-align: center; color: white; font-size:25px;'><span style='font-weight: bold; font-family: century-gothic';></span>Solutions Scope</p>", unsafe_allow_html=True)
        st.selectbox("", ["Select Application", "Conversational AI"], key='application')
        
        # Model Selection
        st.selectbox("", ['LLM Models', "meta-llama/llama-2-70b-chat", "meta-llama/llama-2-13b-chat", "meta-llama/llama-2-7b-chat"], key='text_llmmodel')
        st.selectbox("", ['Features', "Observability/Tracing with Arize","Evaluation (LLM as a Judge)","Human in the Feedback Loop","Guardrails","Prompt Caching","MCP Integration"], key='text_framework')
        st.selectbox("", ["Library Used", "Streamlit", "Crew AI", "IBM Watsonx"], key='text1')
        st.selectbox("", ["IBM Cloud Services Used", "Watsonx Embedding", "Watsonx Vector Store", "Watsonx Retriever"], key='text2')
        
        # Reset button
        st.markdown("#### ")
        href = """<form action="#">
                <input type="submit" value="Clear/Reset"/>
                </form>"""
        st.sidebar.markdown(href, unsafe_allow_html=True)
        
        # Display deployment info
        st.markdown("#")
        st.markdown("<p style='text-align: center; color: White; font-size:20px;'>Build & Deployed on<span style='font-weight: bold'></span></p>", unsafe_allow_html=True)
        s1, s2 = st.columns((4,4))
        with s1:
            st.markdown("### ")
            st.image('image/IBM_Cloud_logo.png')
        with s2:    
            st.markdown("### ")
            st.image("image/st-logo.png")

def display_chat_interface(conversation_manager):
    """Display the chat interface with user type selection and message history"""
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # User type selection
    user_type = st.radio(
        "Select User Type:",
        ["Member", "Employer"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1,4,1])
    
    with col2:
        # Chat history container
        response_container = st.container()
        
        # User input container
        container = st.container()
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                user_input = st.text_input(
                    "Ask your question:",
                    placeholder="How can I help you with your CalPERS benefits?",
                    key='input'
                )
                submit_button = st.form_submit_button(label='Send')
                st.markdown("</div>", unsafe_allow_html=True)
            
            if submit_button and user_input:
                try:
                    
                    response_guardian = guardian.detect(
                        text=user_input,   # required
                        detectors=detectors # optional
                    )
                    print("Response from guardian - ",response_guardian)
                    
                    # Process user input
                    if len(response_guardian["detections"]) > 0:
                        st.write(response_guardian)
                        response = "Detected harmful/guardrails content. Please rephrase your question."
                        eval_results = {}
                    else:
                        response, eval_results = conversation_manager.process_query(user_input, user_type)
                    
                    # Update session state
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(response)
                    st.session_state['classifications'].append(eval_results['classifications'])
                    
                    logger.info(f"Processed query: {user_input}")
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
        
        # Display chat history
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    # Display user message
                    message(
                        st.session_state["past"][i],
                        is_user=True,
                        key=str(i) + '_user',
                        avatar_style="big-smile"
                    )
                    
                    # Display assistant message
                    message(
                        st.session_state["generated"][i],
                        key=str(i),
                        avatar_style="thumbs"
                    )
                    
                    # Display classification results if available
                    if i < len(st.session_state['classifications']):
                        with st.expander("Evaluation Results"):
                            classifications = st.session_state['classifications'][i]
                            
                            # Relevance classifications
                            st.markdown("### Relevance Classifications")
                            st.table(
                                classifications['relevance']
                            )
                            
                            # Hallucination classifications
                            st.markdown("### Hallucination Classifications")
                            st.table(
                                classifications['hallucination']
                            )
                            
                            # Toxicity classifications
                            st.markdown("### Toxicity Classifications")
                            st.table(
                                classifications['toxicity']
                            )

def main():
    """Main application entry point"""
    setup_page()
    initialize_session_state()
    display_header()
    setup_sidebar()
    
    # Display chat interface
    display_chat_interface(conversation_manager)

if __name__ == "__main__":
    main() 