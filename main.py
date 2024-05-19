import os
import streamlit as st
import asyncio
import time
from langchain_google_genai  import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import langchain_helper as lh
from st_pages import Page , show_pages, add_page_title
from langchain_community.llms  import HuggingFaceEndpoint
from langchain_nvidia_ai_endpoints import ChatNVIDIA

st.set_page_config(initial_sidebar_state="expanded" )

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=st.secrets["GOOGLE_API_KEY"],model="models/embedding-001")
# llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9,max_output_tokens=500, google_api_key=st.secrets["GOOGLE_API_KEY"])
llm = HuggingFaceEndpoint(repo_id = "mistralai/Mistral-7B-Instruct-v0.2", 
                        max_length=500, token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])
show_pages(
   [
      Page("main.py", "Home", "🏠"),
      Page("page2.py", "Upload urls or PDF", "📄"),
      Page("page3.py", "Search from Uploaded Data", "🔍"),
      Page("page4.py", "Search from Financial Agents", "🔍")
   ]
)
st.title("Welcome to FinGuru: News Research Tool 📈")



file_path = "faiss_store"

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.9,max_output_tokens=500, google_api_key=st.secrets["GOOGLE_API_KEY"])
# llm = HuggingFaceEndpoint(repo_id = "mistralai/Mistral-7B-Instruct-v0.2", 
#                         max_length=500, token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])
# llm = Ollama(model="gemma")
# llm = ChatCohere(model="command-r-plus", max_tokens=256, temperature=0.75, cohere_api_key=st.secrets["COHERE_API_KEY"])
# llm = ChatNVIDIA(model="meta/llama2-70b", temperature=0.75, nvidia_api_key=st.secrets["NVIDIA_API_KEY"])

prompt_template = """
You are a research assistant.  you answer user queries based on the context provided, and do not make anything by yourself. if you don't know, then just say provided information not given in urls somthing like that,
Only return the  answer in formatted manner use bold,numbers like  below.


Context: {context}

Question: {question}

Answer:
"""

prompt_template2 = """
you are finance assistant, you can answer only  financial and stock related  queries no other queries, you should also provide the sources of the information.
always go with agents search and 
return  the answer and provide the sources of the information in well formatted like bold font numbering pointing. 

Question: {question}

Answer :
Sources :

If the topic is not related to finance, declare it directly and do not proceed with the agent search.
"""


st.markdown("""

FinGuru is a news research tool that processes and analyzes news articles from given URLs and PDF. It leverages LangChain, Google embeddings, and Streamlit to provide insights and answers based on the content of the articles.

## 🎯 Features

- Fetch and parse news articles from URLs Or parse data from given pdf
- Split articles into manageable chunks
- Create embeddings for the text using GoogleEmbedding Model
- Store embeddings in a FAISS index for efficient retrieval
- Query the processed data to get answers and sources

## 🏗️ How It's Built

- Python 3.7+
- Streamlit
- LangChain
- Google API Key
- GOOGLE_CSE_ID

## Used LLM

`google gemini-pro`

## AWS Architecture            
            """)

st.image('./images/Frame.png')

st.markdown("# Equity Research Analysis ")

st.image(['./images/image.png', './images/image-1.png'])
st.markdown("""
            ### Tech Architecture

    - Issue 1 : Copy pasting article in ChatGPt is tedious
    - Issue 2 : We need an aggregate knowledge base
         
            """)

st.image(['./images/image-2.png', './images/image-3.png'])
st.markdown("### Revenue of apple")
st.image("./images/image-5.png")
st.markdown("### calories in apple")
st.image("./images/image-4.png")   
st.markdown("""
            `Semantic search` 
            
            ## Vector Database""")
st.image("images/image-6.png")
st.markdown("## Agents")
st.image("images/image-7.png")

st.markdown("""
            ### Used Agents

`Wikipedia`
`Google Search`
`Google Finance`
`duckduckGo search `

# 🚀 Getting Started

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/mihirh19/news_research_tool_Equity-Research-Analysis-.git
cd news_research_tool_Equity-Research-Analysis-
```

### 2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install the required packages:

```bash
   pip install -r requirements.txt
```

## Setup

1. First, you need to set up the proper API keys and environment variables. To set it up, create the GOOGLE_API_KEY in the Google Cloud credential console (https://console.cloud.google.com/apis/credentials) and a GOOGLE_CSE_ID using the Programmable Search Engine (https://programmablesearchengine.google.com/controlpanel/create). Next, it is good to follow the instructions found here.

2. create api key on https://serpapi.com/

### 3. Create a file named `secrets.toml` in the `.streamlit` directory with the following content:

```toml
GOOGLE_API_KEY = "your-google-api-key"
GOOGLE_CSE_ID = "your-cse-id"
SERP_API_KEY ="your-"
```

## Running the Application

```bash
streamlit run app.py
```

## Usage

1.  Open the Streamlit application in your browser.
2.  Select options From dropdown Menu in the sidebar
3.  For URL :
    - Enter the number of URLs you want to process in the sidebar.
    - Provide the URLs for the news articles.
    - Click on "Process URLs" to fetch and analyze the articles.
4.  For pdf
    - Upload a PDF.
    - Click on "process Pdf" to analyze the PDF.
5.  Enter a query in the text input box and click "Submit" to get answers based on the processed data.

### You can also use the advance google search for financial questions.

## Example 1 URL :

1.  enter 3 as number of urls
2.  provide following urls:
    1. https://www.moneycontrol.com/news/business/tata-motors-to-use-new-1-billion-plant-to-make-jaguar-land-rover-cars-report-12666941.html
    2. https://www.moneycontrol.com/news/business/stocks/tata-motors-stock-jumps-x-after-robust-jlr-sales-brokerages-bullish-12603201.html
    3. https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-1188-sharekhan-12411611.html
3.  Click "Process URLs" to start processing.
4.  Enter a query like `what is the target price of tata motors ?` and click `Submit` to get the answer.

## Example 2 PDF :

1. link Upload the Given PDF
2. Click "Process PDF" to start processing.
3. Enter a query like `what is the yoy change of revenue of tata motors ? `and click `Submit` to get answer.

## Author

👤 **Mihir Hadavani**

- Twitter: [@mihirh21](https://twitter.com/mihirh21)
- Github: [@mihirh19](https://github.com/mihirh19)
- LinkedIn: [@mihir-hadavani-996263232](https://linkedin.com/in/mihir-hadavani-996263232)

## Show your support

Give a ⭐️ if this project helped you!

            
            """)
