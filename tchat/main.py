from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory #FileChatMessageHistory, ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()

chat = ChatOpenAI()

# memory_key is same message key we use for prompt. We set return messages to true to set response to objects not just string
memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages", 
    return_messages=True,
    llm=chat
)

# chat prompt with memory added using messages key added above
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

#  create chain
chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    # verbose=True
)

# start prompt below
while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])