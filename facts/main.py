from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

# create embeddings
embeddings = OpenAIEmbeddings()

# split file into chunks
# find 200 characters then next new line after 200 character
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

# load file
loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter)

#  create chroma store
#  this reaches out to OpenAi and creates embeddings straight away for docs so costs money!
#  will create a directory emb
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

# store in vector store
results = db.similarity_search(
    "What is an interesting fact about the English language?",
    k=2
)

for result in results:
    print("\n")
    print(result.page_content)
    print("\n")
