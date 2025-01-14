import os
import json
import re
import time
from uuid import uuid4
from tqdm import tqdm
from urllib.request import quote, unquote
from bs4 import BeautifulSoup
import requests
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from torch.cuda.amp import autocast

# Configuration settings
BATCH_SIZE = 1000
CHUNK_SIZE_PARENT = 1500
CHUNK_OVERLAP_PARENT = 150
CHUNK_SIZE_CHILD = 600
CHUNK_OVERLAP_CHILD = 100
TOP_K_SEARCH = 10
TOP_K_RERANKER = 10
COUNT_PARAPHRASE = 3


# Utility functions for cleaning links, searching the internet, and fetching text
def clean_links(links, excluded_domains):
    excluded_domains_pattern = '|'.join(re.escape(domain) for domain in excluded_domains)
    cleaned_links = []
    seen_links = set()

    for link in links:
        if not re.search(r'\b(' + excluded_domains_pattern + r')\b', link):
            if link not in seen_links:
                cleaned_links.append(link)
                seen_links.add(link)

    return cleaned_links


def search_query_in_internet(query=None, url=None):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/130.0.2849.80"}

    params = {'hl': 'ru', 'gl': 'ru'}

    if query is not None:
        query += " site:.ru"
        encoded_query = quote(query)

    if url is None:
        url = f"https://www.google.com/search?q={encoded_query}"

    response = requests.get(url=url, headers=headers, verify=False)
    response.encoding = response.apparent_encoding

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        exit()

    return response


def extract_links_from_response(response, excluded_domains, top_k=5):
    soup = BeautifulSoup(response.content, "html.parser")
    all_links = [a.get("href") for a in soup.find_all("a", href=True)]

    external_links = []
    for link in all_links:
        if link.startswith("/url?q=") or link.startswith("http"):
            url_part = link.split('&')[0]
            actual_url = url_part.split("/url?q=")[-1]
            actual_url = unquote(actual_url)

            if actual_url.startswith("http"):
                external_links.append(actual_url)

    cleaned_links = clean_links(external_links, excluded_domains)

    return cleaned_links[:top_k]


def get_text(url):
    response = search_query_in_internet(url=url)
    soup = BeautifulSoup(response.text, "html.parser")
    raw_text = soup.text
    return re.sub(r'[\n]+', '\n', raw_text.replace('\xa0', ' '))


def internet_search(query, k=2):
    excluded_domains = ['google', 'gstatic']
    response = search_query_in_internet(query=query)
    links = extract_links_from_response(response, excluded_domains, top_k=k)

    all_texts = []
    for link in tqdm(links, desc="Fetching links"):
        time.sleep(2)
        all_texts.append(get_text(url=link))

    return all_texts


# Language model class for generating paraphrases and answers
class LanguageModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    def generate_paraphrases(self, query, n=3):
        prompt = f"Generate {n} paraphrases for the following query: {query}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, do_sample=True, max_new_tokens=128)
        paraphrases = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return paraphrases

    def generate_answer(self, question, context):
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with autocast():
            outputs = self.model.generate(**inputs, do_sample=True, max_new_tokens=128)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


# Data processing class for splitting documents
class DataProcessor:
    def __init__(self, chunk_size_parent, chunk_overlap_parent, chunk_size_child, chunk_overlap_child):
        self.splitter_parent = RecursiveCharacterTextSplitter(chunk_size=chunk_size_parent,
                                                              chunk_overlap=chunk_overlap_parent)
        self.splitter_child = RecursiveCharacterTextSplitter(chunk_size=chunk_size_child,
                                                             chunk_overlap=chunk_overlap_child)

    def process_documents(self, texts):
        documents = []
        for text in texts:
            parents_chunks = self.splitter_parent.split_text(text)
            for parent_chunk in parents_chunks:
                children_chunks = self.splitter_child.split_text(parent_chunk)
                for chunk in children_chunks:
                    documents.append(
                        Document(page_content=chunk, id=str(uuid4()), metadata={"parent_chunk": parent_chunk}))
        return documents


# Main function for executing the workflow
def main():
    # Initialize components
    model = LanguageModel(model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    processor = DataProcessor(CHUNK_SIZE_PARENT, CHUNK_OVERLAP_PARENT, CHUNK_SIZE_CHILD, CHUNK_OVERLAP_CHILD)
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                                            model_kwargs={"device": "cuda"})
    client = Chroma(persist_directory="chroma_db_2", embedding_function=embedding_model)

    # Load and process documents
    with open('/kaggle/input/dataset7/ru_wiki_person.txt', 'r', encoding="utf8") as f:
        articles = f.read().split('\n\n')
    documents = processor.process_documents(articles)

    # Add documents to database
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Adding documents"):
        batch = documents[i:i + BATCH_SIZE]
        client.add_documents(batch)

    # Answer generation
    with open("/kaggle/input/dataset/questions.txt", "r", encoding="utf-8") as file:
        questions = file.readlines()

    answers = []
    for question in questions:
        paraphrases = model.generate_paraphrases(question.strip())
        context = internet_search(paraphrases[0])
        answer = model.generate_answer(question.strip(), context)

        start_index = answer.find('Answer:') + len('Answer: ')
        end_index = answer.find('\n', start_index)
        extracted_answer = answer[start_index:end_index].strip()
        answers.append({"question": question.strip(), "answer": extracted_answer})


if __name__ == "__main__":
    main()
