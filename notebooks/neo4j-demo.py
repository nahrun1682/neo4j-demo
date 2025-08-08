import os
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# .envファイルから環境変数を読み込む
load_dotenv()

# 1. Neo4j driver
URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
INDEX_NAME = os.getenv("NEO4J_INDEX_NAME", "document-embeddings")

try:
    # Connect to Neo4j database
    driver = GraphDatabase.driver(URI, auth=AUTH)
    print(f"✅ Neo4j connection established (Database: {DATABASE})")

    # 2. Retriever
    # Create Embedder object, needed to convert the user question (text) to a vector
    embedder = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    )
    print(f"✅ Embeddings model initialized: {embedder.model}")

    # Initialize the retriever
    retriever = VectorRetriever(driver, INDEX_NAME, embedder)
    print(f"✅ Vector retriever initialized with index: {INDEX_NAME} (Database: {DATABASE})")

    # 3. LLM
    # Note: the OPENAI_API_KEY must be in the env vars
    llm = OpenAILLM(
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o"), 
        model_params={"temperature": 0}
    )
    print(f"✅ LLM initialized: {llm.model_name}")

    # Initialize the RAG pipeline
    rag = GraphRAG(retriever=retriever, llm=llm)
    print("✅ GraphRAG pipeline initialized")

    # Query the graph
    query_text = "How do I do similarity search in Neo4j?"
    print(f"\n🔍 Querying: {query_text}")
    
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
    print("\n📝 Response:")
    print(response.answer)
    
    # Close the driver
    driver.close()
    print("\n✅ Neo4j connection closed")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure Neo4j is running and the connection details are correct.")