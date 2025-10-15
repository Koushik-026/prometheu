import streamlit as st
import requests
import xml.etree.ElementTree as ET
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler
from neo4j import GraphDatabase
import json
import re

# --- Custom Callback Handler for a Clean, Live Streaming UI ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.text = ""

    def on_agent_action(self, action, **kwargs):
        self.text += f"**Thought:** {action.log.strip()}\n\n"
        self.text += f"*Using tool: `{action.tool}` with input: `{action.tool_input}`*\n\n"
        self.container.markdown(self.text, unsafe_allow_html=True)

    def on_tool_end(self, output, *, name, **kwargs):
        if name == "PubMedSearch":
            self.text += f"**Observation:** Successfully retrieved research papers from PubMed.\n\n"
        else:
            self.text += f"**Observation:** {output}\n\n"
        self.container.markdown(self.text, unsafe_allow_html=True)

# --- Functions to check if secrets are set ---
def check_secrets():
    required_secrets = [
        "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
        "PUBMED_API_KEY", "GROQ_API_KEY"
    ]
    return all(secret in st.secrets for secret in required_secrets)

# --- Core Functions (search_pubmed, write_to_graph, query_knowledge_graph) ---
def search_pubmed(query: str) -> str:
    search_params = { "db": "pubmed", "term": query, "retmax": 2, "retmode": "xml", "sort": "relevance", "api_key": st.secrets["PUBMED_API_KEY"] }
    try:
        search_response = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=search_params)
        search_response.raise_for_status()
        root = ET.fromstring(search_response.content)
        id_list = root.find("IdList")
        if id_list is None: return "No papers found."
        pmids = [id_elem.text for id_elem in id_list.findall("Id")]
        if not pmids: return "No papers found."
        fetch_params = { "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract", "api_key": st.secrets["PUBMED_API_KEY"] }
        fetch_response = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=fetch_params)
        fetch_response.raise_for_status()
        fetch_root = ET.fromstring(fetch_response.content)
        papers_details = []
        for article in fetch_root.findall(".//PubmedArticle"):
            details = {'title': 'N/A', 'abstract': 'N/A'}
            title_elem = article.find(".//ArticleTitle")
            if title_elem is not None and title_elem.text: details['title'] = title_elem.text.strip()
            abstract_elem = article.find(".//Abstract/AbstractText")
            if abstract_elem is not None and abstract_elem.text: details['abstract'] = abstract_elem.text.strip()
            papers_details.append(details)
        if not papers_details: return "Found papers, but could not extract details."
        return json.dumps(papers_details)
    except Exception as e: return f"Error during PubMed search: {e}"

def write_to_graph(content: str) -> str:
    URI = st.secrets["NEO4J_URI"]
    AUTH = (st.secrets["NEO4J_USERNAME"], st.secrets["NEO4J_PASSWORD"])
    groq_api_key = st.secrets["GROQ_API_KEY"]
    
    extraction_llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)
    prompt = f"""From the JSON data below, extract key scientific entities and their relationships as a list of triplets in the format [ENTITY_1, RELATIONSHIP, ENTITY_2]. You MUST respond with ONLY a valid JSON list of lists. Example: [["entity1", "relationship", "entity2"]]"""
    response = None
    try:
        response = extraction_llm.invoke(prompt)
        json_match = re.search(r'\[\s*\[.*\]\s*\]', response.content, re.DOTALL)
        if not json_match: return f"Could not find valid JSON. Raw response: {response.content}"
        cleaned_response = json_match.group(0)
        triplets = json.loads(cleaned_response)
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with driver.session() as session:
                for triplet in triplets:
                    if isinstance(triplet, list) and len(triplet) == 3:
                        e1, rel, e2 = map(str, triplet)
                        safe_rel = rel.upper().replace(" ", "_").replace("-", "_")
                        query = f"MERGE (a:Entity {{name: $e1}}) MERGE (b:Entity {{name: $e2}}) MERGE (a)-[:`{safe_rel}`]->(b)"
                        session.run(query, e1=e1, e2=e2)
        return f"Successfully added {len(triplets)} facts to the knowledge graph."
    except Exception as e:
        raw_response = response.content if response else "No response from LLM."
        return f"An error occurred while writing to graph: {e}. Raw response: {raw_response}"

def query_knowledge_graph(query: str) -> str:
    URI = st.secrets["NEO4J_URI"]
    AUTH = (st.secrets["NEO4J_USERNAME"], st.secrets["NEO4J_PASSWORD"])
    groq_api_key = st.secrets["GROQ_API_KEY"]

    extraction_llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)
    prompt = f"""From the user's question below, identify the two main scientific concepts. Return these concepts as a JSON list. Example: ["diabetes", "Alzheimer's disease"]\n\nQuestion: "{query}" """
    try:
        response = extraction_llm.invoke(prompt)
        json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if not json_match: return "Could not identify two distinct concepts in the query. Please rephrase."
        cleaned_response = json_match.group(0)
        entities = json.loads(cleaned_response)
        if not isinstance(entities, list) or len(entities) != 2: return "Could not identify two distinct concepts in the query. Please rephrase."
        entity1, entity2 = str(entities[0]).lower(), str(entities[1]).lower()
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with driver.session() as session:
                cypher_query = """MATCH p=(e1:Entity)-[*..4]-(e2:Entity) WHERE toLower(e1.name) CONTAINS $entity1 AND toLower(e2.name) CONTAINS $entity2 RETURN p LIMIT 1"""
                result = session.run(cypher_query, entity1=entity1, entity2=entity2)
                path = result.single()
                if path:
                    nodes = [n['name'] for n in path['p'].nodes]
                    path_str = " -> ".join(nodes)
                    return f"Found a connection in memory: A path exists: {path_str}."
                else:
                    return "No direct connection found in memory between these concepts."
    except Exception as e:
        return f"Error querying the knowledge graph: {e}"

# --- Streamlit App UI ---
st.set_page_config(page_title="Project Prometheus", page_icon="💡", layout="wide")

with st.sidebar:
    st.title("About Project Prometheus")
    st.markdown("""
    **Project Prometheus** is an autonomous AI research agent designed to uncover novel connections in scientific literature. 
    Enter a question, and watch as the agent works through its process in real-time.
    """)

st.title("Project Prometheus")
st.markdown("### Your AI Research Assistant with Memory")

if check_secrets():
    search_tool = Tool(name="PubMedSearch", func=search_pubmed, description="Use to search for new scientific papers on PubMed.")
    memory_writer_tool = Tool(name="WriteToMemory", func=write_to_graph, description="Use to save key findings from text into the knowledge graph. The input for this tool MUST be the raw JSON output from the PubMedSearch tool.")
    memory_reader_tool = Tool(name="QueryKnowledgeGraph", func=query_knowledge_graph, description="Use to check your memory for existing connections between concepts. Input is the original user question.")
    tools = [memory_reader_tool, search_tool, memory_writer_tool]
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=st.secrets["GROQ_API_KEY"])
    
    agent_prompt_template = """(Your detailed agent prompt from before goes here)"""
    
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, agent_kwargs={"input_variables": ["input", "agent_scratchpad"], "template": agent_prompt_template}, verbose=True, max_iterations=12, early_stopping_method="generate", handle_parsing_errors=True )

    with st.form("search_form"):
        topic = st.text_input("Ask a question to find connections", placeholder="e.g., How does gut health relate to mental health?")
        submitted = st.form_submit_button("Ask Agent")

    if submitted and topic:
        output_container = st.container()
        with output_container:
            with st.expander("Show Agent's Thought Process", expanded=True):
                thought_placeholder = st.empty()
            st.markdown("---")
            answer_placeholder = st.empty()

        streamlit_callback = StreamlitCallbackHandler(thought_placeholder)
        
        result = agent.invoke(
            {"input": topic},
            {"callbacks": [streamlit_callback]}
        )
        response = result['output']
        
        with answer_placeholder.container():
            st.success("Final Answer:")
            st.markdown(response)
else:
    st.error("Project secrets are not configured. Please follow the deployment instructions.")