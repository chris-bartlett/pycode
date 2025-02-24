from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from dotenv import load_dotenv

from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool


load_dotenv()

# agent_scratchpad is like a simplifed form of memory tracking conversation almost back and forth with tool requests and answers (original message, function call, and answer)
#  for one agent_executor only. No memeory between different agent_executors
#  can add memory and preserve the final ai message that is returned from the final function call along with the initial human message sent.
chat = ChatOpenAI()

tables = list_tables()

# system message is to tell chat gpt to use describe tool not to just guess!
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI that has access to a SQLite DB. \n"
            f"The database has tables of : {tables}\n"
            "Do not make any assumptions about what tables exist "
            "or what columns exist. Instead, use the 'describe_tables' function"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

# return messages as objects
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

tools = [run_query_tool, describe_tables_tool, write_report_tool]

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
    tools=tools,
    memory=memory
)

# agent_executor("How many users are in the database?")
#  agent_executor("How many users have provided a shipping address?")
# agent_executor("Summarize the top 5 most popular products. Write the results to a report file.?")

# # testing memory
# agent_executor("How many orders were there? Write the results to a report file?")
# agent_executor("Repeat the exact same process for users?")
