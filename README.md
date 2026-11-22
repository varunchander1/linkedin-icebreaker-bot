# LinkedIn Icebreaker Bot

An AI-powered tool that generates personalized icebreakers and conversation starters based on LinkedIn profiles. It uses OpenAI models and LlamaIndex to index a person's profile and answer questions about it.

## Project Overview

Imagine you're heading to a big networking event, surrounded by potential employers and industry leaders. You want to make a great first impression, but you're struggling to come up with more than the usual, "What do you do?"

This bot does the research for you. Give it a LinkedIn profile and it produces three interesting facts about the person's career or education, then lets you ask follow-up questions about them.

## Features

- Extract LinkedIn profile data from a sample profile (mock data) or the ProxyCurl API (see note below)
- Split and index the profile with LlamaIndex and OpenAI embeddings (`text-embedding-3-small`)
- Generate three interesting facts about a person's career or education
- Answer specific questions about the profile, grounded only in its data
- Choose between OpenAI chat models (`gpt-5.4-mini`, `gpt-4o-mini`)
- Use it through a Streamlit web UI or a command-line interface

> **Note:** Proxycurl, the API used for live LinkedIn lookups, [permanently shut down in July 2025](https://nubela.co/blog/goodbye-proxycurl/). The integration is kept in `data_extraction.py` to show the design, but live lookups no longer work. Use **mock data** (`--mock` on the command line, or **Use Mock Data** in the web UI) to run the bot. The extraction module could be adapted to another profile-data provider.

## How It Works

1. **Extract** – `data_extraction.py` fetches profile JSON from ProxyCurl (or a mock JSON file) and strips empty and unneeded fields.
2. **Split & index** – `data_processing.py` turns the JSON into a LlamaIndex `Document`, splits it into chunks (`CHUNK_SIZE = 400`), and builds an in-memory `VectorStoreIndex`.
3. **Query** – `query_engine.py` retrieves the most relevant chunks (`SIMILARITY_TOP_K = 7`) and fills the prompt templates from `config.py` to generate facts or answer questions.

## Project Structure

```
LinkedInProfile-Parser-LlamaIndex/
├── README.md
├── architecture.md            # Detailed architecture and data flow
├── requirements.txt           # Dependencies
├── .env.example               # Template for your OpenAI API key
├── config.py                  # Model settings, query settings, prompt templates
├── modules/
│   ├── __init__.py
│   ├── data_extraction.py     # LinkedIn profile data extraction (ProxyCurl / mock)
│   ├── data_processing.py     # Data splitting, indexing, embedding verification
│   ├── llm_interface.py       # OpenAI embedding and chat LLM setup
│   └── query_engine.py        # Initial facts generation and question answering
├── app.py                     # Streamlit web interface
└── main.py                    # Command-line interface
```

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key
- No LinkedIn API key needed: the bot runs on the built-in mock profile

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-openai-api-key
```


### Usage

#### Web Interface (Streamlit)

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (by default http://localhost:8501). To use a different port, add `--server.port 5000`.

In the **Process LinkedIn Profile** tab, tick **Use Mock Data**, pick a model, and click **Process Profile**. Then switch to the **Chat** tab to ask questions.

#### Command Line Interface

`main.py` does not load the `.env` file, so set `OPENAI_API_KEY` in your shell first:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-openai-api-key"
# macOS / Linux
export OPENAI_API_KEY="your-openai-api-key"
```

Then run:

```bash
python main.py --mock                    # Use mock data
python main.py --mock --model gpt-4o-mini  # Use a different model
```

Type `exit`, `quit`, or `bye` to end the chat.

## Configuration

All settings live in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODELS` | `["gpt-5.4-mini", "gpt-4o-mini"]` | Models offered in the Streamlit UI |
| `LLM_MODEL_ID` | `gpt-5.4-mini` | Default chat model |
| `EMBEDDING_MODEL_ID` | `text-embedding-3-small` | Embedding model |
| `SIMILARITY_TOP_K` | `7` | Chunks retrieved per query |
| `CHUNK_SIZE` | `400` | Chunk size used when splitting profile data |
| `MAX_NEW_TOKENS` | `500` | Maximum tokens per response |
| `INITIAL_FACTS_TEMPLATE` / `USER_QUESTION_TEMPLATE` | – | Prompt templates |

Note: GPT-5 models only accept the default temperature, so `llm_interface.py` automatically uses `temperature=1.0` for them.


## Acknowledgments

- OpenAI for the LLM and embedding models
- LlamaIndex for the data indexing and retrieval framework
- ProxyCurl for LinkedIn profile data extraction


---

**Author:** Varun Chander · [GitHub](https://github.com/varunchander1)
