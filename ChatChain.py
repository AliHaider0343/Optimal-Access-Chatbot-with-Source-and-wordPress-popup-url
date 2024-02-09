from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from operator import itemgetter
from typing import Dict, List, Optional, Sequence
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from pydantic import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.runnable import (Runnable, RunnableBranch,
                                       RunnableLambda, RunnableMap)
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import openai
import streamlit as st
os.environ['OPENAI_API_KEY']="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"
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAJjb0Koe8IdFWQB8jwaVTrwelav20wkMY'

embedding_function = OpenAIEmbeddings()
RESPONSE_TEMPLATE = """\

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "I donot have any information about it because it isn't provided in my context i do apologize for in convenience." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

MUST REMEMBER: Do not answer question on your own Must Refer to the Context If there is no relevant information within the context, just say "Sorry for Inconvenice, i dont have any Information about it in my Digital Brain." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.You are a helpful AI Assistant. Respond to the Greeting Messages Properly."""
REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

refrence_docuemnts_sources=[]
class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]

def create_retriever_chain(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()).with_config(
            run_name="CondenseQuestion", )
    conversation_chain = condense_question_chain | retriever

    return RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
            ),
            (
                    RunnableLambda(itemgetter("question")).with_config(
                        run_name="Itemgetter:question"
                    )
                    | retriever
            ).with_config(run_name="RetrievalChainWithNoHistory"),
        ).with_config(run_name="RouteDependingOnChatHistory")

def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        refrence_docuemnts_sources.append({'Context-Information': doc.page_content,
                   'Source Link': doc.metadata['KuratedContent_sourceUrl'],
                    'Word Press Popup Link': str(doc.metadata['KuratedContent_WordpressPopupUrl'])
                                           })
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)

def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history

def create_chain(llm: BaseLanguageModel,retriever: BaseRetriever,) -> Runnable:
    retriever_chain = create_retriever_chain(
            llm,
            retriever,
        ).with_config(run_name="FindDocs")
    _context = RunnableMap(
            {
                "context": retriever_chain | format_docs,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
        ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RESPONSE_TEMPLATE),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
            run_name="GenerateResponse",
        )
    return (
                {
                    "question": RunnableLambda(itemgetter("question")).with_config(
                        run_name="Itemgetter:question"
                    ),
                    "chat_history": RunnableLambda(serialize_history).with_config(
                        run_name="SerializeHistory"
                    ),
                }
                | _context
                | response_synthesizer
        )

def Get_Conversation_chain(user_id,chatbot_id,query,chat_history,model="gpt-4"):
  # Get the absolute path to the current directory
    current_directory = os.getcwd()

  # Specify the persist directory using the absolute path
    persist_directory = os.path.join(current_directory, f"{user_id}-{chatbot_id}-chroma_db")

    chroma_db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    #retriever = chroma_db.as_retriever(search_kwargs=dict(k=3))
    retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7})

    if model=='gemini-pro':
        llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    else:
        llm = ChatOpenAI(
        model=model,
        streaming=True,
        temperature=0,)

    answer_chain = create_chain(
        llm,
        retriever,
    )
    answer = answer_chain.invoke( {"question": query, "chat_history":chat_history})

    return answer,refrence_docuemnts_sources

#Previous Implemenmtation against interface previous implememtation of Chatbot Invoking
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
# from operator import itemgetter
# from typing import Dict, List, Optional, Sequence
# from langchain.schema.embeddings import Embeddings
# from langchain.schema.retriever import BaseRetriever
# from pydantic import BaseModel
# from langchain.schema.language_model import BaseLanguageModel
# from langchain.schema.output_parser import StrOutputParser
# from langchain.schema.messages import AIMessage, HumanMessage
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
# from langchain.schema.runnable import (Runnable, RunnableBranch,
#                                        RunnableLambda, RunnableMap)
# from langchain.schema import Document
# import os
# import openai
# import streamlit as st
# os.environ['OPENAI_API_KEY'] = "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"
# embedding_function = OpenAIEmbeddings()
# RESPONSE_TEMPLATE = """\
#
# Generate a comprehensive and informative answer of 80 words or less for the \
# given question based solely on the provided search results (URL and content). You must \
# only use information from the provided search results. Use an unbiased and \
# journalistic tone. Combine search results together into a coherent answer. Do not \
# repeat text. Cite search results using [${{number}}] notation. Only cite the most \
# relevant results that answer the question accurately. Place these citations at the end \
# of the sentence or paragraph that reference them - do not put them all at the end. If \
# different results refer to different entities within the same name, write separate \
# answers for each entity.
#
# You should use bullet points in your answer for readability. Put citations where they apply
# rather than putting them all at the end.
#
# If there is nothing in the context relevant to the question at hand, just say "I donot have any information about it because it isn't provided in my context i do apologize for in convenience." Don't try to make up an answer.
#
# Anything between the following `context`  html blocks is retrieved from a knowledge \
# bank, not part of the conversation with the user.
#
# <context>
#     {context}
# <context/>
#
# REMEMBER: Donot answer question on your own Must Refer to the Context If there is no relevant information within the context, just say "Sorry for Inconvenice, i dont have any Information about it in my Digital Brain." Don't try to make up an answer. Anything between the preceding 'context' \
# html blocks is retrieved from a knowledge bank, not part of the conversation with the \
# user.\
# Respond to the Greeting Messages Properly.
# """
# REPHRASE_TEMPLATE = """\
# Given the following conversation and a follow up question, rephrase the follow up \
# question to be a standalone question.
# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone Question:"""
#
# refrence_docuemnts_sources=[]
# class ChatRequest(BaseModel):
#     question: str
#     chat_history: Optional[List[Dict[str, str]]]
#
# def create_retriever_chain(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:
#     CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
#     condense_question_chain = (CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()).with_config(
#             run_name="CondenseQuestion", )
#     conversation_chain = condense_question_chain | retriever
#
#     return RunnableBranch(
#             (
#                 RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
#                     run_name="HasChatHistoryCheck"
#                 ),
#                 conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
#             ),
#             (
#                     RunnableLambda(itemgetter("question")).with_config(
#                         run_name="Itemgetter:question"
#                     )
#                     | retriever
#             ).with_config(run_name="RetrievalChainWithNoHistory"),
#         ).with_config(run_name="RouteDependingOnChatHistory")
#
# def format_docs(docs: Sequence[Document]) -> str:
#     formatted_docs = []
#     for i, doc in enumerate(docs):
#         refrence_docuemnts_sources.append({'Context-Information': doc.page_content,
#                    'Soucre Link': doc.metadata['KuratedContent_sourceUrl']
#                    })
#         doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
#         formatted_docs.append(doc_string)
#     return "\n".join(formatted_docs)
#
# def serialize_history(request: ChatRequest):
#     chat_history = request["chat_history"] or []
#     converted_chat_history = []
#     for message in chat_history:
#         if message.get("human") is not None:
#             converted_chat_history.append(HumanMessage(content=message["human"]))
#         if message.get("ai") is not None:
#             converted_chat_history.append(AIMessage(content=message["ai"]))
#     return converted_chat_history
#
# def create_chain(llm: BaseLanguageModel,retriever: BaseRetriever,) -> Runnable:
#     retriever_chain = create_retriever_chain(
#             llm,
#             retriever,
#         ).with_config(run_name="FindDocs")
#     _context = RunnableMap(
#             {
#                 "context": retriever_chain | format_docs,
#                 "question": itemgetter("question"),
#                 "chat_history": itemgetter("chat_history"),
#             }
#         ).with_config(run_name="RetrieveDocs")
#     prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", RESPONSE_TEMPLATE),
#                 MessagesPlaceholder(variable_name="chat_history"),
#                 ("human", "{question}"),
#             ]
#         )
#
#     response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
#             run_name="GenerateResponse",
#         )
#     return (
#                 {
#                     "question": RunnableLambda(itemgetter("question")).with_config(
#                         run_name="Itemgetter:question"
#                     ),
#                     "chat_history": RunnableLambda(serialize_history).with_config(
#                         run_name="SerializeHistory"
#                     ),
#                 }
#                 | _context
#                 | response_synthesizer
#         )
#
# def Get_Conversation_chain(user_id,chatbot_id,model="gpt-4"):
#     chroma_db = Chroma(persist_directory=f"./{user_id}-{chatbot_id}-chroma_db", embedding_function=OpenAIEmbeddings())
#     #retriever = chroma_db.as_retriever(search_kwargs=dict(k=3))
#     retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7})
#     llm = ChatOpenAI(
#         model=model,
#         streaming=True,
#         temperature=0,
#     )
#     answer_chain = create_chain(
#         llm,
#         retriever,
#     )
#     return answer_chain
