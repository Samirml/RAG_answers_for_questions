# RAG_answers_for_questions

## Description
This project implements a Retrieval-Augmented Generation (RAG) system that integrates a language model with a document retrieval system. In this project, we will create an assistant who will be able to answer any questions about the lives of famous personalities. To do this, we will implement semantic knowledge base search, we will add information search on the Internet.

## Main Features

### 1. **Semantic Knowledge Base Search**
 - **Knowledge Base**: Stores information about famous people from various sources such as Wikipedia. We form documents consisting of children chunks, with their parent chunks (larger text segments) stored in the metadata. This structure enhances context retrieval by allowing query substitution from child to parent, providing more comprehensive responses.
 - **Vector Representations**: Uses a model to generate vector representations of texts, enabling semantic search within the database.
 - **Cross-Encoder for Selection**: To ensure the most relevant chunks are selected, a cross-encoder is employed, which evaluates and ranks the candidate documents based on their relevance to the query.

### 2. **Internet Search**
 - **Real-time Search**: Conducts real-time internet searches to retrieve the latest information about famous personalities.
 - **Context Integration**: Extracts and analyzes contextual information from web pages, which is then used to refine and enrich the generated responses.

### 3. **Response Generation**
 - **RAG Approach**: Combines retrieval of information from both the knowledge base and internet sources with deep learning-based text generation.
 - **Generative Model**: Utilizes the `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` model from Hugging Face. This quantized version of Llama 3.1 excels in generating high-quality text in both English and Russian, making it suitable for multilingual query handling.
 - **Natural Language Processing**: Leverages modern language models to generate well-formed and contextually accurate responses based on the retrieved information.

### 4. **Paraphrasing Queries**
 - **A Variety of Formulations**: Generates multiple paraphrases of the original query to enhance the breadth and depth of the search results.
 - **Improved Search Results**: By reformulating queries in different ways, the system increases the likelihood of retrieving the most relevant and comprehensive information.



## Technologies used
- **Python**: Core programming language for the project.
- **BeautifulSoup**: HTML parsing and text extraction.
- **Requests**: For handling HTTP requests and fetching web content.
- **Hugging Face Transformers**: Pre-trained models for text generation and paraphrasing.
- **PyTorch**: Deep learning framework for model inference.
- **LangChain**: For text splitting and document management.
- **Chroma**: Document embedding storage for efficient retrieval.
- **TQDM**: Progress bar for better user feedback during processing.
- **NumPy**: Utility library for numerical operations.
- **UUID**: For generating unique document identifiers.

  

### Workflow Overview

1. **Initialization**:
   - The project initializes with `__init__.py`, setting up the package structure.

2. **Configuration**:
   - Configuration parameters are defined in `config.json` and managed via `config.py`, allowing easy customization of settings such as batch sizes, chunk sizes, and model parameters.

3. **Main Script Execution**:
   - `main.py` is the central script that orchestrates the entire process:
     - **Data Loading**: Reads the dataset containing information about famous personalities.
     - **Document Processing**: Splits large texts into manageable chunks using text splitters.
     - **Semantic Search Setup**: Initializes Chroma for storing vector embeddings and prepares the knowledge base.
     - **Internet Search**: Fetches additional context from online sources.
     - **Response Generation**: Uses the Llama 3.1 model to generate paraphrases and final answers.

4. **Utility Scripts**:
   - `utils/` contains helper functions and scripts that support the main operations, like data parsing, cleaning, and additional processing.

### 5. **Kaggle Integration**
   - `additional_kaggle_script.py` is a fully implemented standalone version of the project designed specifically for Kaggle. It serves as a backup and can be used if the entire main project fails to run in the local environment.
   - This script ensures that key functionalities, such as data processing, semantic search, and response generation, are still operational within Kaggle’s infrastructure, providing flexibility and reliability for users who prefer or need to use Kaggle's resources.

6. **Database Storage**:
   - `chroma_db/` holds the vector database files, enabling efficient semantic search across the knowledge base.

7. **Dependencies**:
   - `requirements.txt` lists all necessary Python packages, ensuring that the environment is set up correctly for smooth execution.


The program was tested on GPU T4 x2 on kaggle platform. 
To speed up performance, we recommend running on a graphics card with a GPU.

## Customizing the Project

You can easily modify the following parameters to adjust the project’s behavior by editing the `config.json` file. This allows for flexible configuration of various aspects of the project, including data processing, model selection, and search parameters.

### Key Configuration Parameters:

**1. Data Processing:**
   - `BATCH_SIZE`: Determines the number of documents processed in each batch. Adjust this for better performance depending on your system's memory capacity.
   - `CHUNK_SIZE_PARENT`: Specifies the size of the larger text chunks (parent) used in the document splitting process.
   - `CHUNK_OVERLAP_PARENT`: Sets the overlap size between parent chunks to ensure context continuity.
   - `CHUNK_SIZE_CHILD`: Defines the size of the smaller text chunks (child) for more granular document segmentation.
   - `CHUNK_OVERLAP_CHILD`: Specifies the overlap size between child chunks for detailed contextual information.

**2. Search and Retrieval:**
   - `TOP_K_SEARCH`: The number of top search results to retrieve during the initial internet search.
   - `TOP_K_RERANKER`: Determines the number of top results reranked by the cross-encoder for relevance.

**3. Paraphrasing:**
   - `COUNT_PARAPHRASE`: The number of paraphrased queries generated to enhance search results and coverage.

**4. Model Selection:**
   - `MODEL_NAME`: The language model used for generating paraphrases and answers. Default is `"hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"`.
   - `EMBEDDING_MODEL_NAME`: The model used for creating vector embeddings of the text. Default is `"intfloat/multilingual-e5-large"`.

**5. Paths and Storage:**
   - `CHROMA_DB_PATH`: The path where the Chroma database is stored, which holds vector embeddings for semantic search.
   - `DOCUMENTS_PATH`: File path for the input dataset containing information about famous personalities (you can upload a file with your documets inside the project and write a path here)
   - `QUESTIONS_PATH`: File path for the questions to be answered by the system  (you can upload a file with your questions inside the project and write a path here)

### How to Customize:

1. **Editing `config.json`:** 
   - Open the `config.json` file in a text editor.
   - Modify the values according to your needs.
   - Save the file to apply the changes.

This flexible configuration allows you to tailor the project to specific requirements, whether you're adjusting processing sizes, changing the search depth, or selecting different models.



## How to Launch the Project

Follow these steps to set up and run the project:

### Prerequisites

Ensure you have the following installed on your system:
- Python 3.7 or higher
- CUDA-compatible GPU (for optimal performance)
- Required Python packages listed in `requirements.txt`

### Step-by-Step Guide

**1. Clone the Repository:**
   ```bash
   git clone https://github.com/Samirml/RAG_answers_for_questions
   cd RAG_answers_for_questions
   ```
**2. Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv env
   source env/bin/activate
   # On Windows use `env\Scripts\activate`
   ```
**3. Install the required packages:**
```bash
pip install -r requirements.txt
```
**4. Launch the Main Script:**
```bash
python main.py
```

### Future Enhancements

> One potential enhancement to this system is the addition of a conversational support feature. By incorporating a dialogue management module, the system could handle multi-turn conversations, allowing users to engage in dynamic and interactive discussions. This would enable a more personalized and engaging user experience, making the assistant capable of understanding context across multiple queries and maintaining coherent conversations.

> Additionally, if the system is designed to handle various input formats (such as text, audio, or even images), it would be beneficial to explore the integration of a multi-agent system. A multi-agent approach would allow the system to efficiently process and respond to diverse inputs by utilizing specialized agents for each input type, enhancing the system’s versatility and adaptability. This would create a more flexible, comprehensive solution that can cater to a wide range of user needs and interaction modes.

With these future improvements, the system could evolve into a fully interactive assistant, capable of answering complex queries and supporting natural, conversational interactions.



## License
The idea was taken from Karpov.Courses 
(https://karpov.courses/deep-learning?_gl=1*gvc6ll*_ga*NDI1MzY4NTU3LjE3MjM5NzU4OTE.*_ga_DZP7KEXCQQ*MTcyNTg3MzAyNi4xMTYuMC4xNzI1ODczMDI2LjYwLjAuMA..).

## Authors and contacts
To contact the author, write to the following email: samiralzgulfx@gmail.com
