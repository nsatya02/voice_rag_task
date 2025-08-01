# Customer Support Call Analysis with Groq and LangChain

This project implements an end-to-end pipeline to analyze customer support calls using Groq's Whisper and Llama3 models combined with LangChain tooling. It transcribes audio calls, diarizes speakers, extracts structured insights (sentiment, tonality, intent), generates a conversation summary, and provides suggested follow-up actions leveraging a retrieval-augmented generation (RAG) system built on local document embeddings.


## Features

- **Audio Transcription**: Uses Groq's Whisper-large-v3 model to transcribe customer support call audio.
- **Speaker Diarization**: Uses Groq's Llama3-8b model to label and separate speakers into "Customer" and "Agent"
- **Structured Insights Extraction**: Extracts customer sentiment, tonality, and intent using a Pydantic schema and LangChain + Groq LLM.
- **Conversation Summary**: Generates a concise summary of the call.
- **Retrieval-Augmented Generation (RAG)**: Uses a local FAISS vector store with Hugging Face embeddings to suggest clear follow-up actions based on company FAQ or policy documents.

## Prerequisites

- Python 3.11+
- A Groq API key with access to Whisper and Llama3 models

## Tech Stack
- Groq API for advanced transcription(STT) and LLM models.
- LangChain for prompt chaining and LLM orchestration.
- HuggingFace for embeddings.
- FAISS for vector similarity search.

## Installation

1. Clone this repository or copy the script files locally.

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Setup Groq API Key
   Obtain your Groq API key at  [Groq](https://console.groq.com/keys) and save it in a .env file of the project 
  ```bash
  GROQ_API_KEY=your_groq_api_key_here
  ```

5. Input Files
   
   Audio file: Place your customer support call audio file in the project folder and update the AUDIO_FILE_PATH variable in the       script.
   
   FAQ/Policy file: Provide a plain text FAQ or policy file (default is "company_faq.txt"). If not found, a dummy FAQ file will be    created automatically.

6. Usage
   Run the main script:
    ```bash
    python rag_task.py
    ```

This project will:

1. Transcribe and diarize the audio file.
2. Extract structured insights (sentiment, tonality, intent).
3. Generate a concise summary of the call.
4. Set up a RAG system based on the FAQ file and suggest a follow-up action.
5. Print the entire analysis report including the diarized transcript, insights, summary, and suggested actions.

**Done By** : Satya Prasad

**Date** : 01-08-2025
