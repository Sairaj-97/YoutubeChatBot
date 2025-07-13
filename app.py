from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

def get_transcript(url:str)-> str:
    """
    Get the transcript of a YouTube video.
    """
    loader = YoutubeLoader.from_youtube_url(url)
    documents = loader.load()
    page_content = documents[0].page_content
    return page_content

def make_chunks(document:str):
    """
    Split the transcript into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(document)
    return chunks

def get_vector_store(chunks:list):
    """
    Create a vector store from the chunks.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_texts(chunks, embeddings)
    return vector_store

@st.cache_resource
def create_vector_store(url:str):
    """
    Creates a vector store from a YouTube video transcript.
    This function is cached to improve performance.
    """
    try:
        st.info("Loading video transcript... 📺")
        transcript=get_transcript(url)
        st.info("Splitting text into manageable chunks... ✂️")
        chunks= make_chunks(transcript)
        st.info("Creating vector store with embeddings... This may take a moment. ✨")
        vector_store = get_vector_store(chunks)
        st.success("Vector store created successfully! Ready to chat. ✅")
        return vector_store
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def get_response(vector_store, query):
    """
    Get a response from the vector store based on the query.
    """
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Define our prompt template
    prompt_template = """
    You are an intelligent assistant that answers questions about a YouTube video.
    Use the following retrieved context from the video transcript to answer the question.
    If you don't know the answer from the context, just say that you don't know.
    Be concise and helpful.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    retrieved_docs = vector_store.similarity_search(query, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    formatted_prompt = prompt.format(context=context_text, question=query)
    
    # Generate the response
    response = llm.invoke(formatted_prompt)
    
    return response.content


st.set_page_config(page_title="Chat with YouTube", page_icon="▶️")
st.title("▶️ Chat with any YouTube Video")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("Video Input")
    youtube_url = st.text_input("Enter YouTube URL here:")
    
    if st.button("Process Video"):
        if youtube_url:
            with st.spinner("Processing..."):
                st.session_state.vector_store = create_vector_store(youtube_url)
                st.session_state.messages = [] 
        else:
            st.warning("Please enter a YouTube URL.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.vector_store:
    if prompt := st.chat_input("Ask a question about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(st.session_state.vector_store, prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please process a YouTube URL in the sidebar to start chatting.")
