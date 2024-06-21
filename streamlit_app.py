import streamlit as st

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain.callbacks.base import BaseCallbackHandler

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
# import os
import sqlite3


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
    count: Annotated[int, operator.add]

class Queries(BaseModel):
    queries: List[str]

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class ewriter():
    def __init__(self, stream_handler):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", 
                                temperature=0,
                                api_key=st.secrets["OPENAI_API_KEY"],
                                callbacks=[stream_handler])
        self.PLAN_PROMPT = ("You are an expert writer tasked with writing a high level outline of a short 3 paragraph essay. "
                            "Write such an outline for the user provided topic. Give the three main headers of an outline of "
                             "the essay along with any relevant notes or instructions for the sections. ")
        self.WRITER_PROMPT = ("You are an essay assistant tasked with writing excellent 3 paragraph essays. "
                              "Generate the best essay possible for the user's request and the initial outline. "
                              "If the user provides critique, respond with a revised version of your previous attempts. "
                              "Utilize all the information below as needed: \n"
                              "------\n"
                              "{content}")
        self.RESEARCH_PLAN_PROMPT = ("You are a researcher charged with providing information that can "
                                     "be used when writing the following essay. Generate a list of search "
                                     "queries that will gather "
                                     "any relevant information. Only generate 3 queries max.")
        self.REFLECTION_PROMPT = ("You are a teacher grading an 3 paragraph essay submission. "
                                  "Generate critique and recommendations for the user's submission. "
                                  "Provide detailed recommendations, including requests for length, depth, style, etc.")
        self.RESEARCH_CRITIQUE_PROMPT = ("You are a researcher charged with providing information that can "
                                         "be used when making any requested revisions (as outlined below). "
                                         "Generate a list of search queries that will gather any relevant information. "
                                         "Only generate 2 queries max.")
        
        self.tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)
        builder.set_entry_point("planner")
        builder.add_conditional_edges(
            "generate", 
            self.should_continue, 
            {END: END, "reflect": "reflect"}
        )
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(
            checkpointer=memory,
            # interrupt_after=['planner', 'generate', 'reflect', 'research_plan', 'research_critique']
        )
        print("Completed ewriter init")
    
    def plan_node(self, state: AgentState):
        print(f"\n\n Starting plan-node\n$$$$$\n{state}\n@@@@@\n")
        messages = [
            SystemMessage(content=self.PLAN_PROMPT), 
            HumanMessage(content=state['task'])
        ]
        response = self.model.invoke(messages)
        msg=f"PLAN_NODE: Plan: {response.content}\n **** for ****\n MSGS: {messages}\n  *\n **\n***\n"
        #print(msg)
        #st.write(msg)
        return {"plan": response.content,
               "lnode": "planner",
                "count": 1,
               }
    def research_plan_node(self, state: AgentState):
        print(f"\n\n Starting research-plan-node\n$$$$$\n{state}\n@@@@@\n")
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])
        content = state['content'] or []  # add to content
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        msg=f"RESEARCH_PLAN_NODE: {content} fr queries: {queries.queries}"
        #print(msg)
        #st.write(msg)
        return {"content": content,
                "queries": queries.queries,
               "lnode": "research_plan",
                "count": 1,
               }
    
    def research_plan_node_without_tavily(self, state: AgentState):
        messages=[
            SystemMessage(content=self.RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ]
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])
        response = self.model.invoke(messages)
        msg=f"RESEARCH_PLAN_NODE: Plan: Queries {queries} for {response.content}\n **** for ****\n MSGS: {messages}\n  *\n **\n***\n"
        print(msg)
        st.write(msg)
        return {"content": response.content,
               "lnode": "research_plan",
                "count": 1,
               }
    
    def generation_node(self, state: AgentState):
        content = "\n\n".join(state['content'] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
        messages = [
            SystemMessage(
                content=self.WRITER_PROMPT.format(content=content)
            ),
            user_message
            ]
        response = self.model.invoke(messages)
        return {
            "draft": response.content, 
            "revision_number": state.get("revision_number", 1) + 1,
            "lnode": "generate",
            "count": 1,
        }
    def reflection_node(self, state: AgentState):
        print(f"\n\n Starting reflectin-node\n$$$$$\n{state}\n@@@@@\n")
        messages = [
            SystemMessage(content=self.REFLECTION_PROMPT), 
            HumanMessage(content=state['draft'])
        ]
        response = self.model.invoke(messages)
        return {"critique": response.content,
               "lnode": "reflect",
                "count": 1,
        }
    def research_critique_node(self, state: AgentState):
        print(f"\n\n Starting research-critique-node\n$$$$$\n{state}\n@@@@@\n")
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique'])
        ])
        content = state['content'] or []
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content,
               "lnode": "research_critique",
                "count": 1,
        }
    def should_continue(self, state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"

def state_printer(state):
    st.sidebar.write(state)

def main():
    st.write('Hello, world!')
    stream_handler = StreamHandler(st.empty())
    system_prompt=st.text_input('system prompt','')
    if system_prompt:
        print("Starting writer")
        abot = ewriter(stream_handler)
        thread = {"configurable": {"thread_id": "1"}}
        for s in abot.graph.stream({
            'task': system_prompt,
            'max_revisions':2,
            "revision_number": 1,
        },thread):
            print(f"\n\nNew msg:\n++++++++\n{s}\n******\n")
            st.write(s)
            #st.sidebar.write(abot.graph.state)
        print("Completed writer Step 1")


if __name__ == '__main__':
    main()