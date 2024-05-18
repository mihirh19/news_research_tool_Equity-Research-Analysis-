import os
import streamlit as st
import pickle
import time
from langchain_google_genai  import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.llms import Ollama
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
import langchain_helper as lh




file_path = "faiss_store"
main_placeholder = st.empty()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9,max_output_tokens=500, google_api_key=st.secrets["GOOGLE_API_KEY"] )
# llm = HuggingFaceEndpoint(repo_id = "mistralai/Mistral-7B-Instruct-v0.2", 
#                         max_length=500, token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])
# llm = Ollama(model="gemma")
# llm = ChatCohere(model="command-r-plus", max_tokens=256, temperature=0.75, cohere_api_key=st.secrets["COHERE_API_KEY"])
# llm = ChatNVIDIA(model="meta/llama2-70b", temperature=0.75, nvidia_api_key=st.secrets["NVIDIA_API_KEY"])
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=st.secrets["GOOGLE_API_KEY"],model="models/embedding-001")
prompt_template = """
You are a research assistant.  you answer user queries based on the context provided, and do not make anything by yourself. if you don't know, then just say provided information not given in urls somthing like that,
Only return the  answer below.


Context: {context}

Question: {question}

Answer:
"""

prompt_template2 = """
you are finance assistant, you can answer only  financial queries no other queries, you can also provide the sources of the information.
if you don't know,then search from agent and return answer
return the summarized answer and sources.

Question: {question}

Answer :
Sources :

"""







st.title("FinGuru: News Research Tool 📈")
st.sidebar.title("News Article URLs")

option = st.sidebar.selectbox(
   "Source of news article",
   ("URLS", "PDFS")
)
urls = []
pdfs =None
if option == "URLS":
   number_of_urls = st.sidebar.number_input(label="Number of URLS",
                                                min_value=0, max_value=20, value=1)
   urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(number_of_urls)]
   process_url_clicked= st.sidebar.button("Process URLs")
   
   if process_url_clicked:
      if all(urls):
         try:
            with st.status("Loading Data...", expanded=True) as status:
               st.text("Data Loading...Started...✅✅✅")
               vectorstore_urls = lh.create_vector_store_from_urls(urls, embeddings)   
               vectorstore_urls.save_local(file_path)
               status.update(label="Processing complete! ✅ ", state='complete', expanded=False)
            time.sleep(2)
         except Exception as e:
               main_placeholder.error(f"An error occurred: {e}")
      else:
         main_placeholder.error("Please enter all URLs.")
if option == "PDFS":
   pdfs = st.sidebar.file_uploader("Upload file", type=["pdf"])
   process_pdfs_clicked = st.sidebar.button("Process PDFS")
   if process_pdfs_clicked:
      if pdfs:
         try:
            with st.status("Loading Data...", expanded=True) as status:
               st.text("Data Loading...Started...✅✅✅")
               vectorstore_urls = lh.create_vector_store_from_pdfs(pdfs, embeddings)   
               vectorstore_urls.save_local(file_path)
               status.update(label="Processing complete! ✅ ", state='complete', expanded=False)
            time.sleep(2)
         except Exception as e:
               main_placeholder.error(f"An error occurred: {e}")
      else:
         main_placeholder.error("Please enter pdf")

query = st.text_input("Question Related To provided URL or PDF: ")
submit = st.button("Submit")
if submit:
   if query:
      try:
         result = None
         if option == "URLS":
            result = lh.response_of_llm_for_url(llm=llm, file_path=file_path, embeddings=embeddings, query=query, prompt_template=prompt_template)
         if option == "PDFS":
            result = lh.response_of_llm_for_pdf(llm=llm, file_path=file_path, embeddings=embeddings, query=query, prompt_template=prompt_template)
         st.header("Answer")
         if option== "URLS":
            st.write(result["answer"])
         if option == "PDFS":
            st.write(result["result"])
         sources = result.get("sources", "")
         if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
               st.write(source)
      except Exception as e:
            st.error(f"An error occurred: {e}")
   else:
      st.error("Please enter a query.")


st.title("Added (google search) agent for any financial search")

query2 = st.text_input("Anything :......")
button2 = st.button('Search')

if button2:
   if query2:
      try:
         result = lh.response_of_llm(llm=llm, query=query2, prompt_template=prompt_template2)
         st.header("Answer")
         st.write(result)
      except Exception as e:
         st.error(f"An error occurred: {e}")
   else:
      st.error("Please enter a query.")