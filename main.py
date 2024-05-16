import os
import streamlit as st
import pickle
import time
import langchain
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_cohere  import CohereEmbeddings
from langchain_community.vectorstores import FAISS


st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")
urls = []
number_of_urls = st.sidebar.number_input(label="Number of Products",
                                                min_value=0, max_value=20, value=1)

for i in range(number_of_urls):
   url = st.sidebar.text_input(f"URL {i+1}")
   urls.append(url)


process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_cohere_embeddings"
main_placeholder = st.empty()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9,max_output_tokens=500, google_api_key=st.secrets["GOOGLE_API_KEY"])
# llm = Ollama(model="llama3", temperature=0.9, max_tokens=500)
embeddings = CohereEmbeddings(cohere_api_key=st.secrets["COHERE_API_KEY"])

if process_url_clicked:
   
   # load data
   loader = WebBaseLoader(
      web_paths=urls
      
   )
   main_placeholder.text("Data Loading...Started...✅✅✅")
   data = loader.load()
   # split data
   text_splitter = RecursiveCharacterTextSplitter(
      separators=['\n\n', '\n', '.', ','],
      chunk_size=1000
   )
   main_placeholder.text("Text Splitter...Started...✅✅✅")
   docs = text_splitter.split_documents(data)
   # create embeddings and save it to FAISS index

   vectorstore_openai = FAISS.from_documents(docs, embeddings)
   main_placeholder.text("Embedding Vector Started Building...✅✅✅")
   time.sleep(2)
   vectorstore_openai.save_local(file_path)
   # vectorstore = vectorstore_openai.as_retriever() 
   
   
query = main_placeholder.text_input("Question: ")
submit = st.button("Submit")
if submit:
   if query:
      vectorIndex = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
      chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
      result = chain.invoke({"question": query}, return_only_outputs=True)
               # result will be a dictionary of this format --> {"answer": "", "sources": [] }
      st.header("Answer")
      st.write(result["answer"])
      sources = result.get("sources", "")
      if sources:
         st.subheader("Sources:")
         sources_list = sources.split("\n")  # Split the sources by newline
         for source in sources_list:
            st.write(source)
