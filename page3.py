import streamlit as st
from main import file_path, prompt_template, llm, embeddings
import langchain_helper as lh
from page2 import option

st.title("Search from Uploaded Data 🔍")

query = st.text_input(" Question Related To provided URL or PDF: ")
submit = st.button("🔍 Search", type='secondary')

if submit:
   if query:
      try:
         result = None
         if st.session_state.get('option') == "URLS":
            result = lh.response_of_llm_for_url(file_path=file_path,embeddings=embeddings, llm=llm, query=query, prompt_template=prompt_template)
         if  st.session_state.get('option') == "PDF":
            result = lh.response_of_llm_for_pdf(file_path=file_path,embeddings=embeddings, llm=llm, query=query, prompt_template=prompt_template)
         st.header("Answer")
         if st.session_state.get('option') == "URLS":
            st.write(result["answer"])
         if st.session_state.get('option')  == "PDF":
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