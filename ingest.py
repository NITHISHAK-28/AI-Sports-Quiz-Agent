import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def initialize_vector_db():
    # Foundational historical data to seed the local vector store
    sports_data = [
        "Cricket: The first official international cricket match was played in 1844 between the United States and Canada.",
        "Cricket: Sachin Tendulkar holds the record for the most runs in international cricket, scoring over 34,000 runs.",
        "Football: The first FIFA World Cup was held in 1930 in Uruguay, and Uruguay won the tournament.",
        "Football: Real Madrid holds the record for the most UEFA Champions League titles.",
        "Tennis: Roger Federer was the first male player to reach 20 Grand Slam singles titles.",
        "Badminton: India won its first-ever Thomas Cup title in 2022 after defeating Indonesia in the final."
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(sports_data)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "./chromadb_store"
    
    print("Embedding and storing sports data into ChromaDB...")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    print("Database successfully initialized!")

if __name__ == "__main__":
    if not os.path.exists("./chromadb_store"):
        initialize_vector_db()
    else:
        print("ChromaDB already exists.")