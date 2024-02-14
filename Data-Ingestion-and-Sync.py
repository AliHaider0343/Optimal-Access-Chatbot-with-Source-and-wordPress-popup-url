import sys
import xml.etree.ElementTree as ET
import os
import requests
import pandas as pd
import re
import sqlite3
import time
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_transformers import (
    LongContextReorder, )
from langchain_core.documents import Document
from datetime import datetime
import numpy as np
import streamlit as st
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
source_column = "KuratedContent_sourceUrl"
metadata_columns = ['Channel_about', 'Channel_keywords', 'Collection_about', 'Collection_keywords', 'File_about',
                    'File_keywords', 'KuratedContent_author', 'KuratedContent_datePublished',
                    'KuratedContent_dateModified', 'KuratedContent_keywords', 'KuratedContent_publisher',
                    'KuratedContent_sourceUrl','KuratedContent_WordpressPopupUrl']
main_content_columns = ['KuratedContent_Description_and_headline']


def download_and_print_xml(url):
    def fix_br_tags(xml_string):
        fixed_xml_string = re.sub(r'<br([^>]*)>', r'<br\1/>', xml_string)
        return fixed_xml_string.replace('&nbsp;', '')
    try:
        response = requests.get(url)
        if response.status_code == 200:
            xml_content = response.content.decode('utf-8')  # Decode using UTF-8
            fixed_xml_content = fix_br_tags(xml_content)
            return fixed_xml_content
        else:
            print(f"Error: Unable to fetch XML. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")


def extract_text_from_element(element):
    try:
        def get_text_from_element(element):
            text = element.text or ''
            for child in element:
                text += get_text_from_element(child)
            return text

        # Extract text content from the provided element
        text_content = get_text_from_element(element)
        return text_content.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
    return None


def parse_xml(xml_text,wordPress_website_link):
    root = ET.fromstring(xml_text)
    data = {
        'about': root.find('.//span[@itemprop="about"]').text,
        'comment': root.find('.//span[@itemprop="comment"]').text,
        'encoding': root.find('.//span[@itemprop="encoding"]').text,
        'publisher': root.find('.//span[@itemprop="publisher"]').text,
        'author': root.find('.//span[@itemprop="author"]').text,
        'keywords': root.find('.//span[@itemprop="keywords"]').text,
        'Channels': []
    }

    for channel in root.findall('.//group'):
        channel_data = {
            'about': channel.find('.//span[@itemprop="about"]').text,
            'comment': channel.find('.//span[@itemprop="comment"]').text,
            'encoding': channel.find('.//span[@itemprop="encoding"]').text,
            'keywords': channel.find('.//span[@itemprop="keywords"]').text,
            'Collections': []}
        for collection in channel.findall('.//page'):
            collection_data = {
                'about': collection.find('.//span[@itemprop="about"]').text,
                'comment': collection.find('.//span[@itemprop="comment"]').text,
                'encoding': collection.find('.//span[@itemprop="encoding"]').text,
                'keywords': collection.find('.//span[@itemprop="keywords"]').text,
                'KuratedContent': []}
            for artical in collection.findall('.//link'):
                articals_data = {
                    'ID': artical.find('.//span[@itemprop="ID"]').text,
                    'sourceUrl': artical.find('.//meta[@itemprop="mainEntityOfPage"]').get('itemid'),
                    'WordpressPopupUrl': str(wordPress_website_link + clean_wordpress_Link(channel_data['about'],collection_data['about'],(artical.find('.//h2[@itemprop="headline"]').text).strip())),
                    'headline': artical.find('.//h2[@itemprop="headline"]').text,
                    'author': artical.find('.//h3[@itemprop="author"]/span[@itemprop="name"]').text,
                    'description': extract_text_from_element(artical.find('.//span[@itemprop="description"]')),
                    'publisher': artical.find('.//div[@itemprop="publisher"]/meta[@itemprop="name"]').get('content'),
                    'datePublished': artical.find('.//meta[@itemprop="datePublished"]').get('content'),
                    'dateModified': artical.find('.//meta[@itemprop="dateModified"]').get('content'),
                    'keywords': artical.find('.//span[@itemprop="keywords"]').text
                }
                collection_data['KuratedContent'].append(articals_data)
            channel_data['Collections'].append(collection_data)
        data['Channels'].append(channel_data)
    return data

def clean_text(text):
    # Use regular expression to remove unwanted characters
    cleaned_text=text.replace("'","")
    cleaned_text = re.sub(r'[^a-zA-Z0-9\'’-“”]', '-', cleaned_text)
    # Replace empty spaces with hyphens
    cleaned_text = cleaned_text.replace(' ', '-')
    # Remove extra hyphens and convert to lowercase
    cleaned_text = re.sub(r'-+', '-', cleaned_text).lower()
    return cleaned_text.strip('-')

def clean_wordpress_Link(channel_name,collection_name,articals_headline):
    channel_name=channel_name.replace(' ', '-')
    collection_name=collection_name.replace(' ', '-').lower()
    articals_headline=clean_text(articals_headline.replace(' ', '-').lower())

    return '/' +channel_name + '/' + collection_name+ '/'+articals_headline
def check_vector_store_exists(vector_store_name):
    vector_store_path = os.path.join(os.getcwd(), vector_store_name)

    try:
        # Using os.path.join ensures a valid path
        return os.path.exists(vector_store_path)
    except Exception as e:
        # Handle any exceptions that might occur during path checking
        print(f"Error checking vector store: {e}")
        return False


def Append_Single_file_Articals(data):
    Single_data_Collection = []
    for channel_data in data['Channels']:
        for collection_data in channel_data['Collections']:
            for articals_data in collection_data['KuratedContent']:
                row = {
                    'KuratedContent_article_id': articals_data['ID'],
                    'KuratedContent_sourceUrl': articals_data['sourceUrl'],
                    'KuratedContent_WordpressPopupUrl': articals_data['WordpressPopupUrl'],
                    'KuratedContent_headline': articals_data['headline'],
                    'KuratedContent_author': articals_data['author'],
                    'KuratedContent_description': articals_data['description'],
                    'KuratedContent_publisher': articals_data['publisher'],
                    'KuratedContent_datePublished': articals_data['datePublished'],
                    'KuratedContent_dateModified': articals_data['dateModified'],
                    'KuratedContent_keywords': articals_data['keywords'],
                    'Collection_about': collection_data['about'],
                    'Collection_comment': collection_data['comment'],
                    'Collection_encoding': collection_data['encoding'],
                    'Collection_keywords': collection_data['keywords'],
                    'Channel_about': channel_data['about'],
                    'Channel_comment': channel_data['comment'],
                    'Channel_encoding': channel_data['encoding'],
                    'Channel_keywords': channel_data['keywords'],
                    'File_about': data['about'],
                    'File_comment': data['comment'],
                    'File_encoding': data['encoding'],
                    'File_publisher': data['publisher'],
                    'File_author': data['author'],
                    'File_keywords': data['keywords']
                }
                Single_data_Collection.append(row)
    return Single_data_Collection

def Collect_Process_Data(parsed_data_against_urls):
    Articals_Collection = []
    for data in parsed_data_against_urls:
        Articals_Collection.extend(Append_Single_file_Articals(data))
    Dataset_csv = pd.DataFrame(Articals_Collection)
    Dataset_csv['KuratedContent_Description_and_headline'] = Dataset_csv['KuratedContent_headline'] + ':' + Dataset_csv[
        'KuratedContent_description']
    columns_to_drop = ['KuratedContent_headline', 'KuratedContent_description', 'KuratedContent_article_id',
                       'Collection_comment', 'Collection_encoding', 'Channel_comment', 'Channel_encoding',
                       'File_comment', 'File_encoding', 'File_author', 'File_publisher']
    Dataset_csv = Dataset_csv.drop(columns_to_drop, axis=1)
    Dataset_csv = Dataset_csv.drop_duplicates()
    # Dataset_csv['KuratedContent_datePublished'] = pd.to_datetime(Dataset_csv['KuratedContent_datePublished'])
    # Dataset_csv = Dataset_csv.sort_values(by='KuratedContent_datePublished', ascending=False)
    # Dataset_csv['KuratedContent_datePublished'] = Dataset_csv['KuratedContent_datePublished'].dt.strftime('%Y-%m-%d %H:%M:%S')
    Dataset_csv.replace({np.nan: "Not Specified"}, inplace=True)
    Dataset_csv.reset_index()
    return Dataset_csv


# def filter_information(stored_latest_artical_date,processed_data):
#     processed_data['KuratedContent_datePublished_dateFormatted']=pd.to_datetime(processed_data['KuratedContent_datePublished'])
#     processed_data = processed_data[processed_data['KuratedContent_datePublished_dateFormatted'] > stored_latest_artical_date]
#     processed_data.drop(['KuratedContent_datePublished_dateFormatted'],axis=1)
#     return processed_data

def filter_information(stored_latest_article_date, processed_data):
    # Convert the 'KuratedContent_dateModified' column to int64
    processed_data['KuratedContent_dateModified'] = processed_data['KuratedContent_dateModified'].astype('int64')
    # Filter based on the condition
    filtered_data = processed_data[processed_data['KuratedContent_dateModified'] > int(stored_latest_article_date)]
    return filtered_data


def Generate_documents_from_dataframe(processed_data):
    documents = []
    for index, row in processed_data.iterrows():
        data = row[main_content_columns[0]]
        metadata = {column: row[column] for column in metadata_columns}
        document = Document(page_content=data, metadata=metadata)
        documents.append(document)
    return documents


def update_documents_to_delete_pervious_versions_of_updated_Data(chroma_db, filtered_source_urls_list):
    documents = chroma_db.get()
    count = 0
    for source in filtered_source_urls_list:
        for document_id, metadata in zip(documents['ids'], documents['metadatas']):
            if str(metadata['KuratedContent_sourceUrl']) == str(source):
                chroma_db.delete([document_id])
                count += 1
    return f'Previous Version of {count} documents got Deleted From Vector Store.'


def Main_Runner_for_Single_user(Urls_for_chatbot,urls_corresponding_wordPress_website_links, user_id, chatbot_id, Sync_period):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    while True:
        parsed_data_against_urls = []
        for url,wordpress_link in zip(Urls_for_chatbot,urls_corresponding_wordPress_website_links):
            parsed_data_against_urls.append(parse_xml(download_and_print_xml(url),wordpress_link))

        processed_data = Collect_Process_Data(parsed_data_against_urls)
        # here check if we have to create new vector store or we might have to update preious
        if check_vector_store_exists(str(f"{user_id}-{chatbot_id}-chroma_db")):
            # Get the absolute path to the current directory
            current_directory = os.getcwd()
            # Specify the persist directory using the absolute path
            persist_directory = os.path.join(current_directory, f"{user_id}-{chatbot_id}-chroma_db")
    
            chroma_db = Chroma(persist_directory=persist_directory,
                               embedding_function=OpenAIEmbeddings())
            documents = chroma_db.get()
            unique_dates = set()
            for metadata in documents['metadatas']:
                unique_dates.add(int(metadata['KuratedContent_dateModified']))
            stored_latest_artical_date = max(list(unique_dates))
            print(stored_latest_artical_date, len(unique_dates), len(documents['metadatas']))
            filtered_data = filter_information(stored_latest_artical_date, processed_data)
            documents = Generate_documents_from_dataframe(filtered_data)
            print(update_documents_to_delete_pervious_versions_of_updated_Data(chroma_db, list(
                filtered_data['KuratedContent_sourceUrl'])))
            texts = text_splitter.split_documents(documents)
            if len(texts) > 0:
                chroma_db.add_documents(texts)
                print(
                    f'Vector Store has been updated with Latest Inforamtion (Only ({len(documents)}) new Articals Embedings has been calculated and Appended to Existing Vector Store)\n')
            else:
                print(
                    'In Specified SYnc Period There isnt any Data updated on Link Hence nothing to update Vector Store')
        else:
            documents = Generate_documents_from_dataframe(processed_data)
            texts = text_splitter.split_documents(documents)
            current_directory = os.getcwd()
            # Specify the persist directory using the absolute path
            persist_directory = os.path.join(current_directory, f"{user_id}-{chatbot_id}-chroma_db")
            chroma_db = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings(),
                                              persist_directory=persist_directory)
            print('\nVector Store has been Created SUccessfully (Base store)')

        if Sync_period is None:
            print('The user Only want to Create a Chat only once on Provided Data without Knowledge Updation.\n')
            break
        else:
            print('Seelping for (', (Sync_period / 3600), ') hours.')
            time.sleep(Sync_period)
            pass
    return

if __name__ == "__main__":
    # Example usage: This script can be scheduled to run periodically or called by the second script
    if len(sys.argv) == 6:
        urls,wordpress_links, user_id,chatbot_id,Sync_period = sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
        Main_Runner_for_Single_user(urls.split(','),wordpress_links.split(','),user_id, chatbot_id, int(Sync_period))
    else:
        print("Usage: python update_database.py arg1 arg2")

