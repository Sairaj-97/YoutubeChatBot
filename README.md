#  Chat with YouTube – A LangChain & Streamlit Application

> Chat with any YouTube video using AI!  
> Paste a YouTube URL and start asking questions about the video content.  
> Powered by **LangChain**, **OpenAI**, and **Streamlit** using a **Retrieval-Augmented Generation (RAG)** pipeline.

---

##  Features

-  **Chat with Any YouTube Video**  
  Provide a YouTube link and start an intelligent conversation with the video content.

-  **RAG Pipeline (Retrieval-Augmented Generation)**  
  Uses a state-of-the-art LangChain RAG setup for accurate, context-aware answers.

-  **Interactive Streamlit UI**  
  Clean and user-friendly interface for seamless interactions.

-  **Caching**  
  Automatically caches processed transcripts to save time and OpenAI API cost.

-  **Secure API Handling**  
  Keep API keys safe using `.env` variables.

---

##  Tech Stack

| Layer       | Technology                |
|-------------|----------------------------|
| **Backend** | Python, LangChain, OpenAI  |
| **Frontend**| Streamlit                  |
| **Vector DB** | ChromaDB (in-memory)     |
| **Embeddings** | OpenAI Embedding Models |
| **LLM**     | OpenAI GPT (e.g., gpt-3.5) |

---

##  Setup & Installation

Follow the steps below to run the project locally.

### 1.  Clone the Repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2.  Create a Virtual Environment
macOS / Linux:
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4.  Set Environment Variables
Create a .env file in the root directory and add your OpenAI API key:
OPENAI_API_KEY="your_openai_api_key_here"


### 5. Running the Application
Start the Streamlit app with:
```
streamlit run app.py
```



##  How It Works: A Deep Dive into the RAG Pipeline

This application operates on a **Retrieval-Augmented Generation (RAG)** architecture.  
This advanced approach allows the Language Model (LLM) to answer questions based on specific information it wasn't originally trained on .In this case, the content of a specific **YouTube video**.

The process is broken down into **five distinct steps**, executed sequentially by the code:

---

### 1. Load Transcript – `get_transcript` function

**What it is:**  
The process begins by fetching the raw text content of the YouTube video.

**How it works:**  
When a user provides a URL, the `get_transcript` function uses the `YoutubeLoader` from the LangChain library.  
This tool connects to YouTube, extracts the available captions or auto-generated transcript for that specific video, and loads it into memory as a single text document.

---

### 2.  Chunking – `make_chunks` function

**What it is:**  
Language Models have a limited "context window," meaning they can only process a certain amount of text at once.  
Since a full video transcript can be very long, it must be broken into smaller pieces.

**How it works:**  
The `make_chunks` function uses LangChain's `RecursiveCharacterTextSplitter` to split the transcript into chunks of about **1000 characters**, with a **200-character overlap** between consecutive chunks.  
This overlap acts as a connecting thread, ensuring that semantic context isn't lost at chunk boundaries.

---

### 3. Embedding & Storage – `get_vector_store` function

**What it is:**  
This is the core of the retrieval setup — converting text chunks into a numerical format that allows semantic searching.

**How it works:**  
The `get_vector_store` function performs two main actions:

- **Embedding:**  
  Uses OpenAI's `OpenAIEmbeddings` model to transform each text chunk into a high-dimensional vector (a list of numbers). These vectors represent the semantic meaning of the text.

- **Storage:**  
  These vectors (along with the original text chunks) are stored in a **Chroma vector database**, optimized for fast similarity search.  
  This database lives **in memory** while the application is running.

---

### 4. Retrieval – `get_response` function

**What it is:**  
Now that we have a searchable knowledge base, this step finds the most relevant transcript chunks for a user’s question.

**How it works:**  
Inside `get_response`, the following line is executed:

```
vector_store.similarity_search(query, k=5)
```

### 5. 🤖 Generation – `get_response` function (continued)

**What it is:**  
This is the final step where the actual answer is formulated and presented to the user.

**How it works:**

- **Augmentation:**  
  The relevant chunks retrieved in the previous step are combined into a single block of text called the **"context"**.

- **Prompting:**  
  A detailed prompt is constructed using a `ChatPromptTemplate`.  
  This prompt tells the AI:  
  > _"You are a helpful assistant. Please answer the following Question using only the information provided in this Context."_

- **Invocation:**  
  The final, augmented prompt (which includes the instructions, retrieved context, and the user's question) is sent to the `ChatOpenAI` model (e.g., GPT-3.5-Turbo).  
  The model processes this information and generates a coherent, contextually accurate answer, which is returned in the chat interface.


  ## 🖼️ App Preview

Here's how the Chat with YouTube app looks in action:

![App Screenshot](assets/screenshot.png)


