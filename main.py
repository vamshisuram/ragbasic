# minimal RAG script
# 
# load docs
# break docs into chunks
# chunk to embedding (numeric presentation)
# store embedding in vector db .. now it's called vector
# query => convert to embedding
# nearest vector search
# retrieve related chunks + prompt
# llm gives final answer

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# load
loader = TextLoader('data/tips.txt')
docs = loader.load();

# split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# chunk to embedding (numeric representation)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# store embedding.. known as vector from here on.
db = FAISS.from_documents(chunks, embedder)

# query
query = "React VDOM"
results = db.similarity_search(query, k=2)

for r in results:
    print("-----")
    print(r.page_content)



