import streamlit as st
import langchain_helper as lh
from main import llm, prompt_template2

st.title("Added (google search) agent for any financial search")
query2 = st.text_input("Anything :......")
button2 = st.button('🔍 Search')


if button2:
   if query2:
      try:
         res = lh.response_of_llm(llm=llm, query=query2, prompt_template=prompt_template2)
         st.header("Answer")
         st.write(res["output"])
      except Exception as e:
         st.error(f"An error occurred: {e}")
   else:
      st.error("Please enter a query.")
      