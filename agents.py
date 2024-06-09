aaa="""

import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

from openai import OpenAI

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

# memory = SqliteSaver.from_conn_string(":memory:")


class AgentState(TypedDict):
    task: str
    lnode: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    queries: List[str]
    revision_number: int
    max_revisions: int

class Agent:
    def __init__(self,system=""):
        self.system = system
        self.client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
        self.messages = []
        if system:
            self.messages.append({"role":"system","content":system})
        st.write(f"System set up with system={self.system} and messages={self.messages}")

    def __call__(self,message):
        self.messages.append({"role":"user","content":message})
        response = self.execute()
        self.messages.append({"role":"system","content":response})
        return response
    
    def execute(self):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        return completion.choices[0].message.content

"""