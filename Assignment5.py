from selenium import webdriver
import time
import streamlit as st
from selenium.webdriver.common.by import By
import json
from pypdf import PdfReader
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from crewai import Agent, Crew, Task, Process
from crewai.tools import tool


llm=ChatOllama(model="llama3.2:latest")
def function1():
    url = "https://remoteok.com/remote-python-jobs"
    browser = webdriver.Chrome()
    browser.get(url)
    time.sleep(3)

    rows = ""
    jobboards = browser.find_element(By.ID, "jobsboard")
    tbody = jobboards.find_element(By.TAG_NAME, "tbody")
    trr = tbody.find_elements(By.TAG_NAME, "tr")
    for tr in trr:
        try:
            title_tag = tr.find_element(By.CLASS_NAME, "company_and_position")
            title = title_tag.find_element(By.TAG_NAME, "h2").text
        except:
            title = ""
        try:
            company = title_tag.find_element(By.TAG_NAME, "h3").text
        except:
            company = ""
        try:
            tags_td = tr.find_element(By.CLASS_NAME, "tags")
            tag_elements = tags_td.find_elements(By.TAG_NAME, "h3")
            tags = [tag.text.strip() for tag in tag_elements if tag.text.strip()]
        except:
            tags = []
        try:
            image_td = tr.find_element(By.CLASS_NAME, "image")
            script_tag = image_td.find_element(By.TAG_NAME, "script")
            job_json = json.loads(script_tag.get_attribute("innerHTML"))
            description = job_json.get("description", "").strip()
        except:
            description = ""

        rows += f"title: {title}, company: {company}, tags: {'|'.join(tags)}, description: {description}\n\n"

    with open('data.csv', 'w', encoding='utf-8') as file:
        file.write(rows)

#function1()

def extract_text(pdf_file): 
    reader=PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text +'\n'
    return text

# tools

@tool("get_resume_skills")
def extract_skills():
    """
    Returns skills from resume
    """
    data=extract_text(pdf_file)

    prompt = (
        "Extract all technical and soft skills mentioned in the following text. "
        "Respond in bullet points, categorized under Technical Skills and Soft Skills.\n\n"
        f"{data}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool("match_companies")
def match_companies():
    """
    Returns companies that match the resume
    """
    with open ('C:/LSQ/Gen AI/codes/day15/data.csv', 'r', encoding="utf-8") as file:
        data = file.read()
    prompt = ( 
        "Given the following text, identify companies that match the skills and experience mentioned. "
        "Respond with a list of company names.\n\n"
        f"{data}")
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool("skills_required_missing_in_resumes")
def skills_required_missing_in_resumes():
    """
    Returns skills required by companies that are missing in resumes
    """
    with open ('C:/LSQ/Gen AI/codes/day15/data.csv', 'r ', encoding="utf-8") as file:
        data = file.read()
        prompt = (
            "Given the following text, identify skills required by companies that are not mentioned in the resume."
            "Respond with a list of skills.\n\n"
            f"{data}")
        response = llm.invoke([HumanMessage(content=prompt)]) 
        return response.content
    

@tool("trendind_skills")
def trendind_skills():
    """
    Returns trending skills in the industry
    """
    with open ('C:/LSQ/Gen AI/codes/day15/data.csv', 'r',encoding="utf-8") as file:
        data = file.read()
    prompt = (
        "Given the following text, identify trending skills in the industry. "
        "Respond with a list of skills.\n\n"
        f"{data}")
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

#agents

# ResumeAnalyzer → Reads and extracts skills from resume.
ResumeAnalyzer=Agent(
    llm="ollama/llama3.2:latest",
    role="Resume Analyzer Agent",
    goal=" Extract skills from resume.",
    backstory=" This agent is designed to read and extract skills from a resume.",
    tool=[extract_skills],
    verbose=True,
    allow_delegation=False
)

JobMatcher= Agent(
    llm="ollama/llama3.2:latest",
    role=" Job Matcher Agent",
    goal=" Match job requirements with resume skills.",
    backstory=" This agent is designed to match job requirements with resume skills.",
    tool=[match_companies],
    verbose=True,
    allow_delegation=False
)

SkillGapAgent= Agent(
    llm="ollama/llama3.2:latest",
    role=" Skill Gap Agent",
    goal=" Identify skills required by companies that are not mentioned in the resume.",
    backstory=" This agent is designed to identify skills required by companies that are not mentioned in the resume.",
    tool=[skills_required_missing_in_resumes],
    verbose=True,
    allow_delegation=False
)

TrendForecaster=Agent(
    llm="ollama/llama3.2:latest",
    role=" Trend Forecaster Agent",
    goal=" Identify trending skills in the industry.",
    backstory=" This agent is designed to identify trending skills in the industry.",
    tool=[trendind_skills],
    verbose=True,
    allow_delegation=False
)

RecommenderAgent = Agent(
    llm="ollama/llama3.2:latest",
    role="Recommender Agent",
    goal="Synthesize the outputs from all other agents and generate a final, actionable report for the user.",
    backstory="This agent reviews the extracted skills, matched companies, skill gaps, and trending skills, then creates a clear and concise report with recommendations for the user.",
    tool=[],  # will take data from other agents
    verbose=True,
    allow_delegation=False
)

# Tasks
Extract_resume_skills = Task(
    description=" Extract skills from resume. {query}.",
    expected_output=" skills",
    agent=ResumeAnalyzer
)

Find_best_job_matches = Task(
    description=" Find best job matches for the extracted skills. {query}.",
    expected_output=" job matches",
    agent=JobMatcher
)

Detect_skill_gaps = Task(
    description=" Detect skills required by companies that are not mentioned in the resume. {query}.",
    expected_output=" skill gaps",
    agent= SkillGapAgent
)
Forecast_top_trending_skills= Task(
    description=" Forecast top trending skills in the industry. {query}.",
    expected_output=" top trending skills",
    agent= TrendForecaster
)

Summarize_and_recommend_action_plan = Task(
    description="Summarize plan",
    expected_output=" action plan",
    agent= RecommenderAgent
)


crew = Crew(
    agents=[ResumeAnalyzer,JobMatcher,SkillGapAgent,TrendForecaster,RecommenderAgent],
    tasks=[Extract_resume_skills,Find_best_job_matches,Detect_skill_gaps,Forecast_top_trending_skills,Summarize_and_recommend_action_plan],
    process=Process.sequential
)

with st.sidebar:
    st.subheader("Upload pdf file here:")
    pdf_file=st.file_uploader("Upload PDF files", type="pdf")

    if pdf_file:
        text1 = extract_text(pdf_file)

question = st.chat_input(" Ask me a question")
if question:
    response = crew.kickoff({"query": question})
    st.subheader(" Response:")
    st.markdown(response)









    