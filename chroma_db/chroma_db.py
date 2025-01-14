import chromadb
from langchain.embeddings import HuggingFaceEmbeddings


class Chroma:
    """
    A class to interact with the Chroma database for storing and retrieving documents.
    """

    def __init__(self, persist_directory, embedding_function):
        """
        Initializes the Chroma client.

        Args:
            persist_directory (str): Directory to persist the Chroma database.
            embedding_function: The embedding function to use for documents.
        """
        self.client = chromadb.Client()
        self.db = self.client.get_or_create_collection("documents", persist_directory=persist_directory)
        self.embedding_function = embedding_function

    def add_documents(self, documents):
        """
        Adds a list of documents to the Chroma database.

        Args:
            documents (list): List of documents to be added.
        """
        ids = [doc.id for doc in documents]
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function.embed_documents(texts)
        self.db.add(ids, texts, embeddings)

    def query(self, query, k=5):
        """
        Queries the Chroma database for the top-k most relevant documents.

        Args:
            query (str): The search query.
            k (int): The number of results to retrieve.

        Returns:
            list: The top-k most relevant documents.
        """
        embedding = self.embedding_function.embed_query(query)
        return self.db.query(embedding, k=k)
