from langchain_ollama import ChatOllama ,OllamaEmbeddings
import streamlit as st
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# load the model
llm=ChatOllama(model="llama3.2:latest")
# function to generate embeddings

# chat prompt template
template="""
You are a helpful recipe assistant. You will be given a question and a context from a text file.
Use the context to answer the question.If the question cannot be answered using the context, say "I don't know".
Question: {question}    
Context: {context}
Answer:      """
    

file="C:/LSQ/Gen AI/GEN AI case study/text.txt"
with open(file,"r",encoding="utf-8")as text_file:
    documents=text_file.read().split("\n\n")
    #print(docs)

def create_embeddings(documents):
    docs=[]
    for doc in documents:
        docs.append(Document(page_content=doc))
        #for chunking
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs=splitter.split_documents(docs)

     # create vector store 
    vector_store=Chroma(embedding_function = OllamaEmbeddings(model="nomic-embed-text"),persist_directory="./chroma_db",
                        collection_name="text_collection")
    
    vector_store.add_documents(docs)
    st.session_state.vector_store=vector_store

def get_answer_of_user_question(question):

    # get the context based on user question
    context=st.session_state.vector_store.search(question,search_type="similarity", k=2)
    # create the prompt 
    prompt_template=ChatPromptTemplate.from_template(template=template)
    prompt=prompt_template.invoke({"question":question,"context":context})
    
    # get the answer from the model
    response=llm.invoke(prompt)
    st.write(response.content)


create_embeddings(documents)
st.markdown("<h1 style='color:blue;'>RECIPES BOT </h1>",unsafe_allow_html=True)
question=st.chat_input("Ask a question about recipe")
if question:
    get_answer_of_user_question(question)


#if text_file :









