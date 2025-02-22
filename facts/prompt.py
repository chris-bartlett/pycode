from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()


chat = ChatOpenAI()

#  set up embeddings score
embeddings = OpenAIEmbeddings()

#  set up Chroma db but don't go off to chat gpt straight away just put in db
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

# set up retriever to get documents, which will call seat similarity_search
retriever = db.as_retriever()

#  build thr chain with retriever and chat model
#  chain type=stuff just stuff everything into the prompt
chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

#  run query
result = chain.run("What is an interesting fact about the English Language?")

print(result)