from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

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



# store in vector store

for doc in docs:
    print(doc.page_content)
    print("\n")
