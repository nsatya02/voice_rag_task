import os
import json
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. SETUP: LOAD API KEY ---
# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Groq API key not found. Please set it in a .env file.")

# Initialize Groq client
client = Groq(api_key=api_key)

# --- 2. TRANSCRIPTION & DIARIZATION ---
def transcribe_and_diarize_audio(audio_file_path: str) -> str:
    """
    Transcribes an audio file using Groq's Whisper API and then uses
    Groq's Llama3 model to diarize the conversation.
    """
    print(f"1. Transcribing audio from: {audio_file_path}...")
    
    # a. Transcription with Groq's Whisper API
    with open(audio_file_path, "rb") as audio_file:
        transcription_response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="text"
        )
    raw_transcript = transcription_response
    print("   ...Transcription complete.")

    # b. Diarization with Groq's Llama3
    print("2. Diarizing transcript (separating speakers)...")
    diarization_prompt = f"""
    The following is a raw transcript from a customer support call.
    Please reformat it by identifying and labeling the speakers as "Customer:" and "Agent:".
    Add line breaks between speakers to make it readable. Do not add any commentary.

    Transcript:
    {raw_transcript}
    """
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are an expert in formatting conversations."},
            {"role": "user", "content": diarization_prompt}
        ]
    )
    diarized_transcript = response.choices[0].message.content
    print("   ...Diarization complete.")
    return diarized_transcript

# --- 3. STRUCTURED INSIGHTS EXTRACTION ---
class CallInsights(BaseModel):
    """Data model for structured insights from a call."""
    sentiment: Literal['Positive', 'Negative', 'Neutral'] = Field(description="Overall sentiment of the customer")
    tonality: Literal['Calm', 'Angry', 'Polite', 'Frustrated'] = Field(description="The primary tonality of the customer's voice")
    intent: Literal['Complaint', 'Query', 'Feedback', 'Order Placement'] = Field(description="The main purpose of the customer's call")

def extract_structured_insights(transcript: str, llm: ChatGroq) -> dict:
    """
    Extracts sentiment, tonality, and intent from the transcript using LangChain and Groq.
    """
    print("3. Extracting structured insights (Sentiment, Tonality, Intent)...")
    
    parser = PydanticOutputParser(pydantic_object=CallInsights)
    
    prompt_template = """
    Analyze the following customer support call transcript.
    Based on the conversation, extract the customer's overall sentiment, their primary tonality, and their main intent.

    {format_instructions}

    Transcript:
    {transcript}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["transcript"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    insights = chain.invoke({"transcript": transcript})
    
    print("   ...Insight extraction complete.")
    return insights.model_dump()

# --- 4. CONVERSATION SUMMARY ---
def generate_summary(transcript: str, llm: ChatGroq) -> str:
    """
    Generates a brief summary of the conversation using Groq.
    """
    print("4. Generating conversation summary...")
    
    prompt_template = """
    Please provide a concise, one-paragraph summary of the following customer support call.
    Focus on the main reason for the call and the final resolution.

    Transcript:
    {transcript}
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    summary = chain.invoke({"transcript": transcript}).content
    
    print("   ...Summary generation complete.")
    return summary

# --- 5. RAG FOR FOLLOW-UP ACTIONS ---
def setup_rag_system(faq_file_path: str, llm: ChatGroq) -> RetrievalQA:
    """
    Sets up the RAG system by loading, splitting, and indexing documents
    using a local Hugging Face embedding model.
    """
    print("5. Setting up RAG system with Hugging Face embeddings...")
    
    # Load documents
    loader = TextLoader(faq_file_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Create embeddings with a local Hugging Face model
    print("   ...Loading local embedding model (this may take a moment on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store embeddings in FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    print("   ...RAG system is ready.")
    return qa_chain

def get_rag_suggestion(rag_chain: RetrievalQA, query: str) -> str:
    """
    Uses the RAG system to suggest a follow-up action.
    """
    print("6. Generating follow-up action with RAG...")
    
    prompt = f"""
    Based on the following customer query, suggest a clear and concise follow-up action for the support agent.
    Use the retrieved context to provide a specific answer or next step.

    Customer Query: "{query}"
    """
    
    result = rag_chain.invoke(prompt)
    print("   ...RAG suggestion complete.")
    return result['result']

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- INPUTS ---
    AUDIO_FILE_PATH = "recording (1).wav" # Replace with your audio file
    FAQ_FILE_PATH = "company_faq.txt"   # Replace with your FAQ/policy document
    
    # Create dummy files if they don't exist for demonstration
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"Error: Audio file '{AUDIO_FILE_PATH}' not found. Please add a sample audio file to run the script.")
        exit()

    if not os.path.exists(FAQ_FILE_PATH):
        print(f"Warning: FAQ file '{FAQ_FILE_PATH}' not found. Creating a dummy file.")
        with open(FAQ_FILE_PATH, "w") as f:
            f.write("""
Q: What is the return policy?
A: You can return any item within 30 days of purchase for a full refund. The item must be unused and in its original packaging.

Q: How do I track my order?
A: Once your order has shipped, you will receive an email with a tracking number. You can use this number on our website's tracking page.

Q: Do you offer international shipping?
A: Yes, we ship to most countries worldwide. Shipping fees and delivery times vary by destination.
            """)
            
    # --- PIPELINE EXECUTION ---
    
    # 1 & 2. Transcribe and Diarize
    diarized_transcript = transcribe_and_diarize_audio(AUDIO_FILE_PATH)
    
    # Initialize the LLM for subsequent steps
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=api_key)
    
    # 3. Extract Insights
    structured_insights = extract_structured_insights(diarized_transcript, llm)
    
    # 4. Generate Summary
    summary = generate_summary(diarized_transcript, llm)
    
    # 5. Setup and Run RAG
    rag_chain = setup_rag_system(FAQ_FILE_PATH, llm)
    
    # Create a query for the RAG system based on the call's intent and summary
    rag_query = f"The customer's intent was '{structured_insights['intent']}'. Summary: {summary}"
    print(rag_query)
    rag_suggestion = get_rag_suggestion(rag_chain, rag_query)
    
    # --- FINAL OUTPUT ---
    print("\n" + "="*50)
    print("           CUSTOMER CALL ANALYSIS REPORT")
    print("="*50 + "\n")
    
    print("--- DIARIZED TRANSCRIPT ---")
    print(diarized_transcript + "\n")
    
    print("--- STRUCTURED INSIGHTS ---")
    print(json.dumps(structured_insights, indent=2) + "\n")
    
    print("--- CONVERSATION SUMMARY ---")
    print(summary + "\n")
    
    print("--- SUGGESTED FOLLOW-UP ACTION (from RAG) ---")
    print(rag_suggestion + "\n")
    
    print("="*50)
    print("             END OF REPORT")
    print("="*50 + "\n")