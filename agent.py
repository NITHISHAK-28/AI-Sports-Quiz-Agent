from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

def get_quiz_agent():
    # 1. Connect to ChromaDB Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory="./chromadb_store", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    def local_knowledge_tool(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([d.page_content for d in docs])
        
    # 2. Configure Live Web Search
    search_wrapper = TavilySearchAPIWrapper()
    web_search_tool = TavilySearchResults(api_wrapper=search_wrapper)
    
    # 3. Build Agent Executor Chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    system_prompt = """You are an expert sports quiz generator agent. Your job is to create highly accurate multiple-choice quizzes.
    You have access to a web search tool to find recent information. Ground your facts strictly.
    
    Always format your final answer exactly like the example below:
    
    Sport: [Sport Name]
    Difficulty: [Difficulty Level]
    
    Question 1: [Text]
    A. [Option]
    B. [Option]
    C. [Option]
    D. [Option]
    Correct Answer: [Letter]
    Explanation: [Brief ground fact summary]
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    tools = [web_search_tool]
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
