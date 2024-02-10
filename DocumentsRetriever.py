from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.document_transformers import (LongContextReorder, )
from langchain_community.vectorstores import Chroma
import streamlit as st
import pandas as pd

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
def get_size_and_item(path):
    # If it's a file, return its size and "file"
    if os.path.isfile(path):
        return os.path.getsize(path), "file"
    # If it's a directory, sum up the sizes of all files in the directory recursively and return "directory"
    elif os.path.isdir(path):
        total_size = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                file_path = os.path.join(root, f)
                total_size += os.path.getsize(file_path)
        return total_size, "directory"
    else:
        return 0, None

def list_contents_with_size():
    # Get the list of all items (files and directories) in the current directory
    all_contents = os.listdir()

    # Create an empty dictionary to store the results
    result_dict = {}

    # Loop through each item in the list
    for item in all_contents:
        item_path = os.path.join(os.getcwd(), item)
        size, item_type = get_size_and_item(item_path)
        result_dict[item] = (size, item_type)

    return result_dict
    
def retrive_context(query, user_id, chatbot_ID, meta_data=None):
    # Get the absolute path to the current directory
    current_directory = os.getcwd()
    # Specify the persist directory using the absolute path
    persist_directory = os.path.join(current_directory, f"{user_id}-{chatbot_ID}-chroma_db")
    chroma_db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})
    docs = retriever.get_relevant_documents(query)
    contexts = []
    contexts.append(list_contents_with_size())

    for document in docs:
        context = {'Context-Information': document.page_content,
                   'Source Link': document.metadata['KuratedContent_sourceUrl'],
                   'WordPress Popup Link': document.metadata['KuratedContent_WordpressPopupUrl']
                   }
        contexts.append(context)
    return contexts
