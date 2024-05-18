import os
import streamlit as st
import pickle
import time
import langchain
from langchain_google_genai  import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.llms import Ollama
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
# from langchain_cohere  import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate


st.title("FinGuru: News Research Tool 📈")
st.sidebar.title("News Article URLs")
urls = []
number_of_urls = st.sidebar.number_input(label="Number of URLS",
                                                min_value=0, max_value=20, value=1)
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(number_of_urls)]
# pdfs = st.sidebar.file_uploader("Upload files", type=["pdf"])

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_cohere_embeddings"
main_placeholder = st.empty()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9,max_output_tokens=500, google_api_key=st.secrets["GOOGLE_API_KEY"] )
# llm = HuggingFaceEndpoint(repo_id = "mistralai/Mistral-7B-Instruct-v0.2", 
#                         max_length=500, token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])
# llm = Ollama(model="gemma")
# llm = ChatCohere(model="command-r-plus", max_tokens=256, temperature=0.75, cohere_api_key=st.secrets["COHERE_API_KEY"])
# llm = ChatNVIDIA(model="meta/llama2-70b", temperature=0.75, nvidia_api_key=st.secrets["NVIDIA_API_KEY"])
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=st.secrets["GOOGLE_API_KEY"],model="models/embedding-001")

if process_url_clicked:
   if all(urls):
      try:
   # load data
         loader = WebBaseLoader(
            web_paths=urls
            
         )
         # pdf_loader = PyPDFLoader(pdfs)
         with st.status("Loading Data...", expanded=True) as status:
            st.text("Data Loading...Started...✅✅✅")
            data = loader.load()
            # pdf_data = pdf_loader.load_and_split()
            # split data
            text_splitter = RecursiveCharacterTextSplitter(
               separators=['\n\n', '\n', '.', ','],
               chunk_size=1000
            )
            st.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)
            # create embeddings and save it to FAISS index
            
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            # vectorstore_openai = chroma.from_documents(docs, embeddings)
            st.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)
            vectorstore_openai.save_local(file_path)
            status.update(label="Processing complete! ✅ ", state='complete', expanded=False)
         time.sleep(2)
         
      except Exception as e:
            main_placeholder.error(f"An error occurred: {e}")
   else:
      main_placeholder.error("Please enter all URLs.")
   # vectorstore = vectorstore_openai.as_retriever() 
   


prompt_template = """
You are a research assistant.  you answer user queries based on the context provided, and do not make anything by yourself. if you don't know, then just say provided information not given in urls somthing like that,
Only return the  answer below.


Context: {context}

Question: {question}

Answer:
"""
query = st.text_input("Question: ")
submit = st.button("Submit")
if submit:
   if query:
      try:
         vectorIndex = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
         # vectorIndex = chroma.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
         retriever = vectorIndex.as_retriever()
         relevant_docs = retriever.invoke(query)
         context = " ".join([doc.page_content for doc in relevant_docs])
         if not context.strip():
            st.error("No relevant context found for the given question.")
         else :
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
            formatted_prompt = prompt.format(context=context, question=query)

            
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            result = chain.invoke({"question": formatted_prompt}, return_only_outputs=True)
                     # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])
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
