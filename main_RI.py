import streamlit as st
from langchain_helper_RI import get_few_shot_db_chain

st.title("Corporate Strategy Planner: Database Q&A :writing_hand:") #spiral_note_pad writing_hand
#st.set_page_config(page_title= "Corporate Strategy Planner: Database Q&A", page_icon=":bar_chart:", layout="wide")

question = st.text_input("Question: ")

if question:
    pass
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)