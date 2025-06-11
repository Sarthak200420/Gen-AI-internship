from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import mysql.connector

# Connect to MySQL
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Sarthak@20@04",
    database="college"
)

# Create the model
model = ChatOllama(model="gemma2:2b")

# Prompt for SQL generation (updated version)
query_template = """
You are a MySQL developer, writing MySQL queries.
Use the given schema and answer users' questions.
Important: In the 'grades' and 'attendance' tables, the student_id is an INTEGER that references the 'id' column in the 'students' table.
If a question refers to a student by their external student_id (like 'S1001'), join the 'students' table to get the correct internal id.
Return only the raw SQL query. Do NOT use triple backticks, markdown, or any explanation.

Question: {question}
Schema: {schema}

Query:
"""

# Prompt for result summarization
result_template = """
You are a database summarizer.
Given the user's question, generated query, and results found, you need to format the results accordingly.

Examples:

Question: what is the toppers name?
Query: SELECT name FROM students WHERE student_id = (SELECT student_id FROM grades ORDER BY grade LIMIT 1);
Results: [('John',)]
Formatted result: The toppers name is John.

Question: {question}
Query: {query}
Results: {results}

Print only the formatted result.
"""

# Schema for reference
schema = """
CREATE TABLE students (
    id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    student_id VARCHAR(20) UNIQUE NOT NULL,
    department VARCHAR(50),
    year INT
);

CREATE TABLE professors (
    id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(50)
);

CREATE TABLE courses (
    id INT PRIMARY KEY,
    course_code VARCHAR(20) UNIQUE NOT NULL,
    course_name VARCHAR(100) NOT NULL,
    professor_id INT,
    FOREIGN KEY (professor_id) REFERENCES professors(id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);

CREATE TABLE grades (
    id INT PRIMARY KEY,
    student_id INT,
    course_id INT,
    grade CHAR(1) CHECK (grade IN ('A', 'B', 'C', 'D', 'F')),
    FOREIGN KEY (student_id) REFERENCES students(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE attendance (
    id INT PRIMARY KEY,
    student_id INT,
    course_id INT,
    date DATE,
    present BOOLEAN,
    FOREIGN KEY (student_id) REFERENCES students(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
"""

# Generate query using LLM
def generate_query(question):
    prompt = ChatPromptTemplate.from_template(query_template).invoke({
        "question": question,
        "schema": schema
    })
    result = model.invoke(prompt)
    query = result.content.strip()
    if query.startswith("```sql") or query.startswith("```"):
        query = query.replace("```sql", "").replace("```", "").strip()
    return query

# Execute SQL query
def execute_query(query):
    print(f"query = {query}")
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchall()

# Format results using LLM
def format_result(question, query, results):
    prompt = ChatPromptTemplate.from_template(result_template).invoke({
        "question": question,
        "query": query,
        "results": results
    })
    result = model.invoke(prompt)
    print(result.content.strip())

# Main loop
if __name__ == "__main__":
    try:
        while True:
            question = input("\nEnter your question (or type 'exit'): ")
            if question.lower() == "exit":
                break

            try:
                query = generate_query(question)
                results = execute_query(query)
                format_result(question, query, results)
            except Exception as e:
                print(f"Error during execution: {e}")

    finally:
        connection.close()
        print("Database connection closed.")
