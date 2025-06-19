import streamlit as st
import pandas as pd
from crewai import Agent, Crew, Task, Process
from crewai.tools import tool

# Tools
@tool("get_weather_info")
def get_weather_info():
    """
    Returns weather info as a string.
    """
    df = pd.read_csv(r"C:/LSQ/Gen AI/codes/day_13/weather_data.csv")
    return df.to_string(index=False)

@tool("get_aqi_info")
def get_aqi_info():
    """
    Returns AQI info as a string.
    """
    df = pd.read_csv(r"C:/LSQ/Gen AI/codes/day_13/aqi_data.csv")
    return df.to_string(index=False)

# Agents
weather_agent = Agent(
    llm="ollama/llama3.2:latest",
    role="Weather Agent",
    goal="Provide weather information",
    backstory="I am a weather agent, I can provide you with current weather information.",
    tool=[get_weather_info],
    verbose=True,
    allow_delegation=False
)

aqi_agent = Agent(
    llm="ollama/llama3.2:latest",
    role="AQI Agent",
    goal="Provide AQI information",
    backstory="I am an AQI agent, I can provide you with current AQI information.",
    tool=[get_aqi_info],
    verbose=True,
    allow_delegation=False
)

# Tasks
weather_task = Task(
    description="I want to get the current weather information from the weather_data.csv file. User query: {query}.",
    expected_output="A string summary of current weather info.",
    agent=weather_agent
)

aqi_task = Task(
    description="I want to get the current AQI information from the aqi_data.csv file. User query: {query}.",
    expected_output="A string summary of current AQI info.",
    agent=aqi_agent
)
crew = Crew(
    agents=[weather_agent, aqi_agent],
    tasks=[weather_task, aqi_task],
    process=Process.sequential
)


st.header(" Weather and AQI Information")
st.write("This app helps you get current weather and AQI information from provided data files.")
question = st.chat_input("Ask me a question about the weather or AQI")

if question:
    with st.spinner("Crew is executing..."):

        #query_lower = question.lower()

        response = crew.kickoff({"query": question})

        st.subheader(" Response:")
        #st.write(f"```text\n{response}\n```")
        st.write(response)
