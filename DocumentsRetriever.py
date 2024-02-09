from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.document_transformers import (LongContextReorder, )
from langchain_community.vectorstores import Chroma
os.environ['OPENAI_API_KEY']="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"
import pandas as pd

def retrive_context(query, user_id, chatbot_ID, meta_data=None):
    chroma_db = Chroma(persist_directory=f"./{user_id}-{chatbot_ID}-chroma_db", embedding_function=OpenAIEmbeddings())
    retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})
    docs = retriever.get_relevant_documents(query)
    contexts = []
    for document in docs:
        context = {'Context-Information': document.page_content,
                   'Source Link': document.metadata['KuratedContent_sourceUrl'],
                   'WordPress Popup Link': document.metadata['KuratedContent_WordpressPopupUrl']
                   }
        contexts.append(context)
    return contexts