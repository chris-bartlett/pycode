from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv

from tools.sql import run_query_tool


load_dotenv()

# agent_scratchpad is like a simplifed form of memory tracking conversation almost back and forth with tool requests and answers (original message, function call, and answer)
chat = ChatOpenAI()
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

tools = [run_query_tool]

# agent is a chain that knows how to do tools
agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

# AgentExecutor takes an agent and runs it over and over until gets a response not to make a function call
# different ways to create agent - can initialize_agent which automatically creates an agent for you where above is manual
agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools
)

agent_executor("How many users are in the database?")
