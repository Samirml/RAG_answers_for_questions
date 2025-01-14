import os
import json
from uuid import uuid4
from tqdm import tqdm
from utils.data_processor import DataProcessor
from utils.language_model import LanguageModel
from utils.search_utils import internet_search
from chromadb.chromadb import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from config import load_config


def main():
    """
    Main function to execute the workflow for processing documents, adding them to the
    Chroma database, and answering questions using the model and web search results.

    1. Loads configuration settings.
    2. Initializes components like the language model, data processor, and Chroma client.
    3. Processes and adds documents to the Chroma database.
    4. Retrieves answers to questions by generating paraphrases and searching the web.
    """
    # Load configuration settings
    config = load_config()

    # Initialize components with config values
    model = LanguageModel(model_name=config["MODEL_NAME"])
    processor = DataProcessor(
        chunk_size_parent=config["CHUNK_SIZE_PARENT"],
        chunk_overlap_parent=config["CHUNK_OVERLAP_PARENT"],
        chunk_size_child=config["CHUNK_SIZE_CHILD"],
        chunk_overlap_child=config["CHUNK_OVERLAP_CHILD"]
    )
    embedding_model = HuggingFaceEmbeddings(model_name=config["EMBEDDING_MODEL_NAME"], model_kwargs={"device": "cuda"})
    client = Chroma(persist_directory=config["CHROMA_DB_PATH"], embedding_function=embedding_model)

    # Load and process documents
    with open(config["DOCUMENTS_PATH"], 'r', encoding="utf8") as f:
        articles = f.read().split('\n\n')
    documents = processor.process_documents(articles)

    # Add documents to the database
    for i in tqdm(range(0, len(documents), config["BATCH_SIZE"]), desc="Adding documents"):
        batch = documents[i:i + config["BATCH_SIZE"]]
        client.add_documents(batch)

    # Answer generation
    with open(config["QUESTIONS_PATH"], "r", encoding="utf-8") as file:
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
