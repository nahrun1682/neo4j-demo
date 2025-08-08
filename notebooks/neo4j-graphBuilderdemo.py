import os
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.embeddings import OpenAIEmbeddings
# Knowledge Graph Builder
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# .envファイルから環境変数を読み込む
load_dotenv()

# 1. Neo4j driver
URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
INDEX_NAME = os.getenv("NEO4J_INDEX_NAME", "document-embeddings")

# Graph Schema Setup
basic_node_labels = ["Object", "Entity", "Group", "Person", "Organization", "Place"]

academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]

medical_node_labels = ["Anatomy", "BiologicalProcess", "Cell", "CellularComponent",
                "CellType", "Condition", "Disease", "Drug",
                    "EffectOrPhenotype", "Exposure", "GeneOrProtein", "Molecule",
                    "MolecularFunction", "Pathway"]

node_labels = basic_node_labels + academic_node_labels + medical_node_labels

# define relationship types
rel_types = ["ACTIVATES", "AFFECTS", "ASSESSES", "ASSOCIATED_WITH", "AUTHORED","BIOMARKER_FOR"]
try:
    # Connect to Neo4j database
    neo4j_driver = GraphDatabase.driver(URI, auth=AUTH)
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


    prompt_template = '''
    You are a medical researcher tasks with extracting information from papers
    and structuring it in a property graph to inform further medical and research Q&A.

    Extract the entities (nodes) and specify their type from the following Input text.
    Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node.


    Return result as JSON using the following format:
    {{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
    "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

    ...

    Use only fhe following nodes and relationships:
    {schema}

    Assign a unique ID (string) to each node, and reuse it to define relationships.
    Do respect the source and target node types for relationship and the relationship direction.

    Do not return any additional information other than the JSON in it.

    Examples:
    {examples}

    Input text:

    {text}
    '''
    
    kg_builder_pdf = SimpleKGPipeline(
    llm=lm,
    driver=neo4j_driver,
    text_splitter=FixedSizeSplitter(chunk_size=500, chunk_overlap=100),
    embedder=embedder,
    entities=node_labels,
    relations=rel_types,
    prompt_template=prompt_template,
    from_pdf=True
)

except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure Neo4j is running and the connection details are correct.")