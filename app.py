import streamlit as st
from streamlit_chat import message
from streamlit_feedback import streamlit_feedback
from dotenv import load_dotenv
import os
from loguru import logger
from src.utils.config import load_config
from src.utils.logging_config import setup_logging
from src.agents.conversation import ConversationManager

# Load environment variables and setup
load_dotenv()
setup_logging()
logger.info("Starting CalPERS Multi-Agent Assistant")

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
        st.selectbox("", ['LLM Framework', "Langchain","Crew AI"], key='text_framework')
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

def display_chat_interface():
    """Display the main chat interface"""

    # Center the user type selection
    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    user_type = st.radio(
        "I am a:",
        ["Member", "Employer"],
        horizontal=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat history container with centered content
    st.markdown("""
        <style>
            .stChatMessage {
                max-width: 800px !important;
                margin: 0 auto !important;
                padding: 1rem !important;
            }
            .stChatMessageContent {
                max-width: 800px !important;
                margin: 0 auto !important;
            }
            .stForm {
                max-width: 800px !important;
                margin: 0 auto !important;
            }
            .stTextInput {
                max-width: 800px !important;
                margin: 0 auto !important;
            }
            .stButton {
                max-width: 800px !important;
                margin: 0 auto !important;
            }
            .stFeedback {
                max-width: 800px !important;
                margin: 0 auto !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    response_container = st.container()
    
    # User input container with centered content
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
                # Process user input
                response = conversation_manager.process_query(user_input, user_type)
                
                # Update session state
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(response)
                
                logger.info(f"Processed query: {user_input}")
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error("I apologize, but I encountered an error. Please try again.")

    # Display chat history with centered messages
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                # Center align user message
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + '_user',
                    avatar_style="big-smile"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Center align assistant message
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                message(
                    st.session_state["generated"][i],
                    key=str(i+55),
                    avatar_style="thumbs"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Center align feedback
                if i != 0:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    feedback = streamlit_feedback(
                        feedback_type="thumbs",
                        optional_text_label="[Optional] Please provide feedback on this response",
                        key=f"thumbs_{i}"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    if feedback:
                        logger.info(f"Received feedback: {feedback}")

def main():
    """Main application entry point"""
    setup_page()
    initialize_session_state()
    display_header()
    setup_sidebar()
    
    # Create a container for the chat interface
    col1,col2,col3 = st.columns((2,8,2))
    with col2:
        st.markdown("""
            <style>
                .stContainer {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 10rem;
                }
            </style>
        """, unsafe_allow_html=True)
        display_chat_interface()

if __name__ == "__main__":
    main() 