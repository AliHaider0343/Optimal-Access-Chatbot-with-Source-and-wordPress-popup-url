from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.document_transformers import (LongContextReorder, )
from langchain_community.vectorstores import Chroma
import streamlit as st
import pandas as pd

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
def retrive_context(query, user_id, chatbot_ID, meta_data=None):
    # Get the absolute path to the current directory
    current_directory = os.getcwd()
    # Specify the persist directory using the absolute path
    persist_directory = os.path.join(current_directory, f"{user_id}-{chatbot_ID}-chroma_db")
    chroma_db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})
    docs = retriever.get_relevant_documents(query)
    contexts = []
    for document in docs:
        context = {'Context-Information': document.page_content,
                   'Source Link': document.metadata['KuratedContent_sourceUrl'],
                   'WordPress Popup Link': str(document.metadata['KuratedContent_WordpressPopupUrl']).lower()
                   }
        contexts.append(context)
    return contexts
