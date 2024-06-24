
# AI Assistant for SQL and Business Queries

This project provides an AI Assistant that helps with SQL query generation and answering business-related questions. It uses the following components:

- **LangChain**: For managing the language model and vector database.
- **Chroma**: As the vector store for document embeddings.
- **Streamlit**: For building the user interface.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/JustinWebconnex/sql-writter-rag
    cd ai-assistant
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the initial data file and place it in the project directory.

## Usage

1. Load the documents from the CSV file and clean them:

    ```python
    documents = load_documents_from_csv(file_path)
    ```

2. Initialize the embedding function using a Sentence Transformer model:

    ```python
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    ```

3. Load or create the Chroma database:

    ```python
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
    else:
        db = Chroma.from_documents(documents, embedding_function, persist_directory=CHROMA_DB_DIR)
        db.persist()
    ```

4. Set up the language model and retriever for retrieval-augmented generation (RAG):

    ```python
    llm = ChatOpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.0)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    ```

5. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

## Application Features

### SQL Query Helper

The SQL Query Helper assists in generating SQL queries based on user-provided context and a predefined set of tables. It uses a prompt template to format the user input and generates the SQL query accordingly.

### Wbx Oracle Assistant

The Wbx Oracle Assistant answers business-related questions using the same underlying technology. It retrieves relevant documents and provides detailed answers based on the user query.

## License

This project is under the Webconnex License.

## Contributing

We welcome contributions to improve the project. Please fork the repository and create a pull request with your changes.

## Contact

For any questions or issues, please open an issue on GitHub or contact us directly.

---

**Information is accurate as of May 21, 2024.**