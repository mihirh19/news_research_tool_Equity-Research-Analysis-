from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_community.utilities import GoogleSearchAPIWrapper, WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_google_genai  import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.llms  import HuggingFaceEndpoint
from langchain_community.tools import DuckDuckGoSearchRun,DuckDuckGoSearchResults


text_splitter = RecursiveCharacterTextSplitter(
               separators=['\n\n', '\n', '.', ','],
               chunk_size=1000
            )
def create_vector_store_from_urls(urls, embeddings):
         loader = WebBaseLoader(
            web_paths=urls
            
         )
         data = loader.load()
         docs = text_splitter.split_documents(data)
         vectorstore_urls = FAISS.from_documents(docs, embeddings)
         return vectorstore_urls
      
def create_vector_store_from_pdfs(pdfs, embeddings):
         loader = PdfReader(pdfs)
         txt =""
         for page in loader.pages:
            txt += page.extract_text()
         
         docs = text_splitter.split_text(txt)
         vectorstore_pdfs = FAISS.from_texts(docs, embeddings)
         return vectorstore_pdfs
      

def response_of_llm_for_url(file_path ,query, prompt_template, embeddings, llm):
   vectorIndex = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
   retriever = vectorIndex.as_retriever()
   relevant_docs = retriever.invoke(query)
   context = " ".join([doc.page_content for doc in relevant_docs])
   
   prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
   formatted_prompt = prompt.format(context=context, question=query)
   chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
   result = chain.invoke({"question": formatted_prompt}, return_only_outputs=True)
   return result

def response_of_llm_for_pdf(file_path,query, prompt_template, embeddings, llm):
   vectorIndex = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
   retriever = vectorIndex.as_retriever()
   relevant_docs = retriever.invoke(query)
   context = " ".join([doc.page_content for doc in relevant_docs])
   prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
   formatted_prompt = prompt.format(context=context, question=query)
   print(formatted_prompt)
   chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
   
   result = chain.invoke({"query": formatted_prompt}, return_only_outputs=True)
   return result

def response_of_llm(llm, query, prompt_template):
   search = GoogleSearchAPIWrapper(google_api_key=st.secrets["GOOGLE_API_KEY"], google_cse_id=st.secrets["GOOGLE_CSE_ID"])
   google_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
   )
   # wiki = load_tools(['wikipedia'], llm=llm)
   wiki = WikipediaAPIWrapper()
   wiki_tool = Tool(
    name="wikipedia",
    description="Search Wikipedia for relevant information.",
    func=wiki.run,
   )
   gfi = GoogleFinanceAPIWrapper(serp_api_key=st.secrets["SERP_API_KEY"])
   gfi_tool = Tool(
    name="google_finance",
    description="Search Google Finance for relevant information.",
    func=gfi.run,
   )
   duck = DuckDuckGoSearchRun(back="news")
   duck_tool = Tool(
    name="duckduckgo",
    description="Search DuckDuckGo for relevant information.",
    func=duck.run,
   )
   
   prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
   
   formatted_prompt = prompt.format(question=query)
   tools = [ duck_tool]

   agent = initialize_agent(llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_iterations=5)
   # agent_executor = AgentExecutor(agent=agent, handle_parsing_errors=True, tools=tools)
   return agent.invoke({"input": formatted_prompt})
   