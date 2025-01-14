from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from uuid import uuid4


class DataProcessor:
    """
    A class to handle document processing, including splitting documents into smaller chunks.

    Attributes:
        splitter_parent (RecursiveCharacterTextSplitter): The splitter for larger parent chunks.
        splitter_child (RecursiveCharacterTextSplitter): The splitter for smaller child chunks.
    """

    def __init__(self, chunk_size_parent, chunk_overlap_parent, chunk_size_child, chunk_overlap_child):
        """
        Initializes the DataProcessor with the given chunking parameters.

        Args:
            chunk_size_parent (int): The size of the parent chunk.
            chunk_overlap_parent (int): The overlap between parent chunks.
            chunk_size_child (int): The size of the child chunk.
            chunk_overlap_child (int): The overlap between child chunks.
        """
        self.splitter_parent = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_parent, chunk_overlap=chunk_overlap_parent
        )
        self.splitter_child = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_child, chunk_overlap=chunk_overlap_child
        )

    def process_documents(self, texts):
        """
        Processes a list of texts into smaller chunks and returns them as Document objects.

        Args:
            texts (list): A list of strings containing the document text.

        Returns:
            list: A list of Document objects containing the chunked text.
        """
        documents = []
        for text in texts:
            parents_chunks = self.splitter_parent.split_text(text)
            for parent_chunk in parents_chunks:
                children_chunks = self.splitter_child.split_text(parent_chunk)
                for chunk in children_chunks:
                    documents.append(Document(page_content=chunk, id=str(uuid4()), metadata={"parent_chunk": parent_chunk}))
        return documents
