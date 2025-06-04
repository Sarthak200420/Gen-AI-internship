import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

llm = ChatOllama(model="llama3", temperature=0.4)


st.markdown("<h1 style='color:blue;'>Career Guidance Assistant</h1>", unsafe_allow_html=True)
st.subheader("Get personalized career advice and insights")

# Role 
selection = st.selectbox("Choose your area of interest:", ["Counselor", "Resume Advisor", "Interview Coach"])


roles = {
    "Counselor": "Provides career advice and guidance to students",
    "Resume Advisor": "Offers feedback on resume content and structure",
    "Interview Coach": "Gives mock interview questions, answers, and tips"
}
qa_history=[]   

prompt = st.chat_input("Enter your query here:")


if prompt:
    st.write("Query:",prompt)
    system_message = SystemMessage(content=roles[selection])
    messages = [
        system_message,
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    qa_history.append(f"Q:{prompt}\nA:{response.content}")
    
    st.markdown("<h2 style='color:green;'>Response from chatbot</h2>", unsafe_allow_html=True)
    st.write(response.content)

#button
if qa_history:
    full_text = "\n\n".join(qa_history)
    st.download_button("Download all questions and answers", full_text, file_name="qa_history.txt")