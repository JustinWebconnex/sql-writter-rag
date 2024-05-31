# import
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
import re

# load the document and split it into chunks
file_path = "/content/sql-writer-finetune - Sheet1.csv"
df = pd.read_csv(file_path)

# Reemplazar NaN con cadenas vacías en las columnas específicas
df = pd.read_csv(file_path)

# Reemplazar NaN con cadenas vacías en las columnas específicas
df['Instruction'] = df['Instruction'].fillna('No Instruction')
df['Input'] = df['Input'].fillna('No Input')
df['Response'] = df['Response'].fillna('No Response')
# Realizar la concatenación
df['page_content'] = df['Instruction'] + df['Input'] + df['Response']

# Convertir metadata a cadena de texto y reemplazar NaN por cadena vacía
df['Metadata'] = df['Metadata'].fillna('No metadata').astype(str)

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

def clean_metadata_string(metadata_str):
    # Eliminar caracteres especiales que puedan causar problemas
    cleaned_metadata_str = re.sub(r'[\[\]\']', '', metadata_str)
    # Convertir a diccionario simple
    metadata_dict = {"description": cleaned_metadata_str} if cleaned_metadata_str else {}
    return metadata_dict

# Crear la lista de documentos
documents = []
for index, row in df.iterrows():
    page_content = row['page_content']
    cleaned_metadata_dict = clean_metadata_string(row['Metadata'])
    documents.append(Document(page_content=page_content, metadata=cleaned_metadata_dict))

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(documents, embedding_function)

import os
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

LLM_OPENAI_GPT35 = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.)
retriever = db.as_retriever(search_kwargs={"k":4})

QUERY_PROMPT_TEMPLATE = """\
Human:
You are an expert SQL writer. Create a SQL query based on the provided context. Only use the following tables to create the query:
- wbx_data_dbt.dim_account(
    id INTEGER,
    name CHARACTER VARYING,
    date_created TIMESTAMP,
    date_updated TIMESTAMP,
    date_deleted TIMESTAMP,
    source CHARACTER VARYING,
    referral_code CHARACTER VARYING,
    contact CHARACTER VARYING,
    email CHARACTER VARYING,
    phone CHARACTER VARYING,
    organization_id INTEGER
)
- wbx_data_dbt.dim_account_managers (
    id INTEGER,
    first_name CHARACTER VARYING,
    last_name CHARACTER VARYING,
    date_created DATE,
    date_deleted DATE
)
- wbx_data_dbt.dim_account_managers_account (
    id NUMERIC,
    account_manager_id INTEGER
)    
- wbx_data_dbt.dim_billing (
    id INTEGER,
    account_id INTEGER,
    product CHARACTER VARYING(11),
    date_created TIMESTAMP WITHOUT TIME ZONE,
    waive_invoices SMALLINT,
    package_id INTEGER,
    package_name CHARACTER VARYING(150),
    date_next_billing TIMESTAMP WITHOUT TIME ZONE
)
- wbx_data_dbt.dim_billing_item (
    id INTEGER,
    account_id INTEGER,
    billing_id INTEGER,
    page_id INTEGER,
    subject_type CHARACTER VARYING(75),
    subject_id BIGINT,
    exclude_from_invoicing SMALLINT,
    pricing_type INTEGER,
    quantity INTEGER,
    total INTEGER,
    currency CHARACTER VARYING(9),
    date_created TIMESTAMP WITHOUT TIME ZONE,
    date_updated TIMESTAMP WITHOUT TIME ZONE
)
- wbx_data_dbt.dim_dates (
    full_date DATE,
    month_day_number INTEGER
);

- wbx_data_dbt.dim_event_end_dates (
    form_id INTEGER,
    event_end TIMESTAMP WITHOUT TIME ZONE,
    event_end_source CHARACTER VARYING(27)
);

- wbx_data_dbt.dim_form (
    id INTEGER,
    account_id INTEGER,
    currency CHARACTER VARYING(9),
    product CHARACTER VARYING(11),
    name CHARACTER VARYING(225),
    date_created TIMESTAMP WITHOUT TIME ZONE,
    event_start TIMESTAMP WITHOUT TIME ZONE,
    event_end TIMESTAMP WITHOUT TIME ZONE
);
- dim_deal
- wbx_data_dbt.dim_gateway (
    id INTEGER,
    type CHARACTER VARYING(150),
    payment_method_provider_id BIGINT,
    gateway_provider_type CHARACTER VARYING(150),
    date_created TIMESTAMP WITHOUT TIME ZONE
);

- wbx_data_dbt.dim_invoice (
    id INTEGER,
    billing_id INTEGER,
    account_id INTEGER,
    package_id BIGINT,
    product CHARACTER VARYING(11),
    package_name CHARACTER VARYING(150),
    amount NUMERIC,
    date_created TIMESTAMP WITHOUT TIME ZONE,
    complete_date TIMESTAMP WITHOUT TIME ZONE,
    start_date TIMESTAMP WITHOUT TIME ZONE,
    end_date TIMESTAMP WITHOUT TIME ZONE,
    billing_date TIMESTAMP WITHOUT TIME ZONE,
    status CHARACTER VARYING(150),
    forwarded_to_invoice_id INTEGER,
    forwarded_from_invoice_id BIGINT
);
- dim_product
- wbx_data_dbt.dim_registrant_data (
    registration_id INTEGER,
    account_id INTEGER,
    form_id INTEGER,
    hash CHARACTER VARYING(168),
    registrant_count INTEGER
);

- wbx_data_dbt.dim_registration_data (
    id INTEGER,
    account_id INTEGER,
    form_id INTEGER,
    hash CHARACTER VARYING(168),
    total NUMERIC,
    status SMALLINT,
    date_created TIMESTAMP WITHOUT TIME ZONE
);

- wbx_data_dbt.dim_ticket_data (
    registration_id INTEGER,
    account_id INTEGER,
    form_id INTEGER,
    hash CHARACTER VARYING(168),
    registration_date DATE,
    last_ticket_date DATE,
    ticket_count INTEGER
);
- wbx_data_dbt.dim_transaction (
    id INTEGER,
    account_id INTEGER,
    form_id INTEGER,
    registration_id INTEGER,
    gateway_id INTEGER,
    transaction_type SMALLINT,
    status INTEGER,
    amount NUMERIC,
    amount_refunded NUMERIC,
    payment_method CHARACTER VARYING(384),
    source_type SMALLINT,
    currency CHARACTER VARYING(9),
    app_fee NUMERIC,
    date_created TIMESTAMP WITHOUT TIME ZONE
);

Look for examples in the {context} on how to use these tables and structure the query. If you do not know how to proceed with a specific part, state that you do not have the context to formulate it, but continue the query as far as possible. For example, say that a WHERE clause is missing if you do not have the context to formulate it.

{context}
Question: {question}
Assistant:
"""
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=retriever, return_source_documents=True,)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)}
)

def main():
    st.title("SQL Query Assistant")

    # Get user input
    query = st.text_area("Enter your query:")

    # Create a button to submit the query
    if st.button("Submit"):
        try:
            response = qa_chain({"query": query})
            st.markdown("**Result:**")
            st.code(response['result'], language="sql")
            st.markdown("**Sources:**")
            st.code(response['source_documents'], language="sql")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()