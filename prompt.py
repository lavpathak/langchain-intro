from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_rertriever import RedundantFilterRetriever
import langchain

langchain.debug = True

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="emb", # this is using the already calculated embeddings from facts.py. That file was only run once
    embedding_function=embeddings # this is same as embedding argument in facts.py but different constructor is used
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("what is an interesting fact about english language?")

print(result)