"""Streamlit web interface for the Icebreaker Bot.

Run with:
    streamlit run app.py --server.port 5000
"""

import sys
import logging
import streamlit as st
from dotenv import load_dotenv

# Load OPENAI_API_KEY (and any other secrets) from a .env file
load_dotenv()

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data, create_vector_database, verify_embeddings
from modules.llm_interface import change_llm_model
from modules.query_engine import generate_initial_facts, answer_user_query
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Available LLM models (OpenAI)
AVAILABLE_MODELS = config.LLM_MODELS

def process_profile(linkedin_url, api_key, use_mock, selected_model):
    """Process a LinkedIn profile and generate initial facts.

    Args:
        linkedin_url: LinkedIn profile URL to process.
        api_key: ProxyCurl API key.
        use_mock: Whether to use mock data.
        selected_model: LLM model to use.

    Returns:
        A tuple of (initial facts about the profile, vector index).

    Raises:
        ValueError: If any processing step fails.
    """
    # Change LLM model if needed
    if selected_model != config.LLM_MODEL_ID:
        change_llm_model(selected_model)

    # Use a default URL for mock data if none provided
    if use_mock and not linkedin_url:
        linkedin_url = "https://www.linkedin.com/in/varun_chander/"

    # Extract profile data
    profile_data = extract_linkedin_profile(
        linkedin_url,
        api_key if not use_mock else None,
        mock=use_mock
    )

    if not profile_data:
        raise ValueError("Failed to retrieve profile data. Please check the URL or API key.")

    # Split data into nodes
    nodes = split_profile_data(profile_data)

    if not nodes:
        raise ValueError("Failed to process profile data into nodes.")

    # Create vector database
    index = create_vector_database(nodes)

    if not index:
        raise ValueError("Failed to create vector database.")

    # Verify embeddings
    if not verify_embeddings(index):
        logger.warning("Some embeddings may be missing or invalid")

    # Generate initial facts
    facts = generate_initial_facts(index)

    return facts, index

def chat_with_profile(index, user_query):
    """Chat with a processed LinkedIn profile.

    Args:
        index: Vector index for the processed profile (None if not loaded).
        user_query: User's question.

    Returns:
        The assistant's reply.
    """
    if index is None:
        return "No profile loaded. Please process a LinkedIn profile first."

    try:
        # Answer the user's query
        response = answer_user_query(index, user_query)
        return response.response

    except Exception as e:
        logger.error(f"Error in chat_with_profile: {e}")
        return f"Error: {str(e)}"

def main():
    """Render the Streamlit interface for the Icebreaker Bot."""
    st.set_page_config(page_title="LinkedIn Icebreaker Bot")

    # Per-browser-session state (Streamlit reruns this script on every interaction)
    st.session_state.setdefault("index", None)
    st.session_state.setdefault("facts", "")
    st.session_state.setdefault("messages", [])

    st.title("LinkedIn Icebreaker Bot")
    st.write("Generate personalized icebreakers and chat about LinkedIn profiles")

    process_tab, chat_tab = st.tabs(["Process LinkedIn Profile", "Chat"])

    with process_tab:
        left, right = st.columns(2)

        with left:
            with st.form("profile_form"):
                linkedin_url = st.text_input(
                    "LinkedIn Profile URL",
                    placeholder="https://www.linkedin.com/in/username/"
                )
                api_key = st.text_input(
                    "ProxyCurl API Key (Leave empty to use mock data)",
                    placeholder="Your ProxyCurl API Key",
                    type="password",
                    value=config.PROXYCURL_API_KEY or ""
                )
                use_mock = st.checkbox("Use Mock Data", value=True)
                selected_model = st.selectbox(
                    "Select LLM Model",
                    AVAILABLE_MODELS,
                    index=AVAILABLE_MODELS.index(config.LLM_MODEL_ID)
                    if config.LLM_MODEL_ID in AVAILABLE_MODELS else 0
                )
                submitted = st.form_submit_button("Process Profile")

            if submitted:
                with st.spinner("Processing profile..."):
                    try:
                        facts, index = process_profile(linkedin_url, api_key, use_mock, selected_model)
                        st.session_state.index = index
                        st.session_state.facts = facts
                        # Start a fresh chat for the new profile
                        st.session_state.messages = []
                    except Exception as e:
                        logger.error(f"Error in process_profile: {e}")
                        st.error(f"Error: {str(e)}")

        with right:
            st.subheader("Initial Facts")
            if st.session_state.facts:
                st.success("Profile processed successfully!")
                st.markdown(f"Here are 3 interesting facts about this person:\n\n{st.session_state.facts}")
            else:
                st.info("Process a profile to see initial facts.")

    with chat_tab:
        st.write("Chat with the processed LinkedIn profile")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # st.chat_input is kept outside the tabs so it pins to the bottom of the page
    user_query = st.chat_input("What is this person's current job title?")
    if user_query and user_query.strip():
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.spinner("Thinking..."):
            reply = chat_with_profile(st.session_state.index, user_query)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

main()
