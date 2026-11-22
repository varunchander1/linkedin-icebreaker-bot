"""Configuration settings for the Icebreaker Bot."""

# OpenAI settings (API key is read from the OPENAI_API_KEY environment variable / .env file)
OPENAI_API_BASE = "https://api.openai.com/v1"
CONTEXT_WINDOW = 128000

# Model settings
LLM_MODELS = ["gpt-5.4-mini", "gpt-4o-mini"]
LLM_MODEL_ID = "gpt-5.4-mini"

EMBEDDING_MODEL_ID = "text-embedding-3-small"

# ProxyCurl API settings
PROXYCURL_API_KEY = ""  # Replace with your API key

# Mock data URL
MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"

# Query settings
SIMILARITY_TOP_K = 7
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
MIN_NEW_TOKENS = 1
TOP_K = 50
TOP_P = 1

# Node settings
CHUNK_SIZE = 400

# LLM prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, using only the information provided in the context. If the answer is not available in the context, say "I don't know. The information is not available on the LinkedIn page."
"""
