import streamlit as st
import time
import langchain_helper as lh
from main import file_path, embeddings
st.title("Upload Data 📄")

option = st.selectbox(
   "Source of news article",
   ("URLS", "PDF")
)
main_placeholder = st.empty()
urls = []
pdfs =None
if option == "URLS":

   number_of_urls = st.number_input(label="Number of URLS",
                                                min_value=0, max_value=20, value=1)
   urls = [st.text_input(f"URL {i+1}") for i in range(number_of_urls)]
   process_url_clicked= st.button("Process URLs")

   if process_url_clicked:
      if all(urls):
         try:
            pro = st.progress(0)
            vectorstore_urls = lh.create_vector_store_from_urls(urls, embeddings)   
            pro.progress(50)
            vectorstore_urls.save_local(file_path)
            pro.progress(100)
            time.sleep(1)
            pro.empty()
            st.session_state.option = "URLS"
         except Exception as e:
            main_placeholder.error(f"An error occurred: {e}")
      else:
         main_placeholder.error("Please enter all URLs.")
if option == "PDF":
   
   pdfs = st.file_uploader("Upload file", type=["pdf"])
   process_pdfs_clicked = st.button("Process PDF")
   if process_pdfs_clicked:
      if pdfs:
         try:
            pro = st.progress(0)
            vectorstore_urls = lh.create_vector_store_from_pdfs(pdfs ,embeddings)   
            pro.progress(50)
            vectorstore_urls.save_local(file_path)
            pro.progress(100)
            time.sleep(1)
            pro.empty()
            st.session_state.update(option="PDF")
         except Exception as e:
               main_placeholder.error(f"An error occurred: {e}")
      else:
         main_placeholder.error("Please enter pdf")