import streamlit as st
from langchain_ollama import ChatOllama
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
import mysql.connector
from sqlalchemy import create_engine

# MySQL credentials
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_DATABASE = "employee"

# Load the model
model = ChatOllama(model="Gemma2:2b")

# Prompt templates
query_template = """
You are a MySQL developer. Write a valid SQL query based only on the given schema.

DO NOT use any columns not listed in the schema.
If the question refers to a non-existent column, ignore it or return a partial result.
You can use multiple tables if needed.

Question: {question}
Schema:
{schema}

Write only the SQL query as output.
"""

result_template = """
You are a helpful assistant.
Given a user's question, the SQL query used, and the results returned from the database,
respond with a direct, concise, and well-formatted natural language answer.

Question: {question}
Query: {query}
Results: {results}

Answer:
"""

st.markdown("<h1 style='color:blue;'>Chat With Me</h1>", unsafe_allow_html=True)

# Sidebar upload
with st.sidebar:
    st.subheader("Upload CSV files")
    csv_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if csv_files:
    # MySQL engine
    engine = create_engine(f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}")
    uploaded_tables = []

    # Upload files and create tables
    for file in csv_files:
        table_name = file.name.split('.')[0]
        df = pd.read_csv(file)
        df.to_sql(table_name, con=engine, index=False, if_exists='replace')
        uploaded_tables.append(table_name)

    # MySQL connection
    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )

    # Extract schema for all tables
    def get_combined_schema():
        full_schema = ""
        cursor = connection.cursor()
        for table in uploaded_tables:
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """, (MYSQL_DATABASE, table))
            columns = cursor.fetchall()
            column_descriptions = ", ".join([f"{col} ({dtype})" for col, dtype in columns])
            full_schema += f"Table `{table}`: {column_descriptions}\n"
        cursor.close()
        return full_schema.strip()

    # Generate SQL query from model
    def generate_query(question, schema):
        prompt_template = ChatPromptTemplate.from_template(template=query_template)
        prompt = prompt_template.invoke({
            "question": question,
            "schema": schema
        })
        response = model.invoke(prompt)
        content = response if isinstance(response, str) else response.content
        content = content.strip()

        # Clean code block markers
        if content.startswith("```sql") or content.startswith("```"):
            content = content.replace("```sql", "").replace("```", "").strip()
        elif content.startswith("'''sql") or content.startswith("'''"):
            content = content.replace("'''sql", "").replace("'''", "").strip()

        return content

    # Execute the SQL query
    def execute_query(query):
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result

    # Format final result using model
    def format_result(question, query, results):
        prompt_template = ChatPromptTemplate.from_template(template=result_template)
        prompt = prompt_template.invoke({
            "question": question,
            "query": query,
            "results": results
        })
        final_result = model.invoke(prompt)
        return final_result if isinstance(final_result, str) else final_result.content

    # Chat input
    question = st.chat_input("Ask a question (can use any uploaded table):")

    if question:
        try:
            full_schema = get_combined_schema()
            query = generate_query(question, full_schema)
            results = execute_query(query)
            final_ans = format_result(question, query, results)
            st.write(final_ans)
        except Exception as e:
            st.error(f"Error: {e}")

    connection.close()
