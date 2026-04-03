from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import os
from getpass import getpass
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

#tavily = TavilyClient(api_key=os.getenv("TAVILI_API_KEY"))
#@tool
#def search(query: str) -> str:
    #"""Search for information using TavilySearch tool
    #ARGS:
        #query: The search query string
    #RETURNS:
        #str: The search results returned by the TavilySearch tool
    #"""
    #print(f"searching for: {query}")
    #return tavily.search(query=query, num_results=1)
class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass()
    
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001", # or "claude-3-opus-20240229"
   
)
search_tool = TavilySearch(max_results=1)
tools = [search_tool]
#tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    """
    Main entry point for the program.
    Prints a hello message and invokes the agent with a prompt about the weather.
    """
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="what is the weather like in the Tehran area today?"
            )
        }
    )
    print(result)

if __name__ == "__main__":
    main()
