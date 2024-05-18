from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.agents.initialize import initialize_agent


text_splitter = RecursiveCharacterTextSplitter(
               separators=['\n\n', '\n', '.', ','],
               chunk_size=1000
            )
def create_vector_store_from_urls(urls,embeddings):
         loader = WebBaseLoader(
            web_paths=urls
            
         )
         data = loader.load()
         docs = text_splitter.split_documents(data)
         vectorstore_urls = FAISS.from_documents(docs, embeddings)
         return vectorstore_urls
      
def create_vector_store_from_pdfs(pdfs,embeddings):
         loader = PdfReader(pdfs)
         txt =""
         for page in loader.pages:
            txt += page.extract_text()
         
         docs = text_splitter.split_text(txt)
         vectorstore_pdfs = FAISS.from_texts(docs, embeddings)
         return vectorstore_pdfs
      

def response_of_llm_for_url(file_path, embeddings, llm ,query, prompt_template):
   vectorIndex = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
   retriever = vectorIndex.as_retriever()
   relevant_docs = retriever.invoke(query)
   context = " ".join([doc.page_content for doc in relevant_docs])
   
   prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
   formatted_prompt = prompt.format(context=context, question=query)
   chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
   result = chain.invoke({"question": formatted_prompt}, return_only_outputs=True)
   return result

def response_of_llm_for_pdf(file_path, embeddings, llm ,query, prompt_template):
   vectorIndex = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
   retriever = vectorIndex.as_retriever()
   relevant_docs = retriever.invoke(query)
   context = " ".join([doc.page_content for doc in relevant_docs])
   prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
   formatted_prompt = prompt.format(context=context, question=query)
   chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
   result = chain.invoke({"query": formatted_prompt}, return_only_outputs=True)

   return result

def response_of_llm(llm, query, prompt_template):
   search = GoogleSearchAPIWrapper(google_api_key=st.secrets["GOOGLE_API_KEY"], google_cse_id=st.secrets["GOOGLE_CSE_ID"])
   tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
   )
   
   prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
   
   formatted_prompt = prompt.format(question=query)
   tools = [tool]
   agent = initialize_agent(llm=llm, tools=tools, agent="zero-shot-react-description", verbose=True)
   print(agent)
   # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
   result = agent.run({"input": formatted_prompt})
   return result
   