from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
from ibm_watsonx_ai.foundation_models.extensions.rag import VectorStore, Retriever
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from loguru import logger
from pydantic import Field
import traceback

from src.utils.config import load_config

class SearchTool(BaseTool):
    """Tool for searching the knowledge base"""
    name: str = "search_knowledge_base"
    description: str = "Search the knowledge base for relevant information. If no relevant information is found, return 'NO_RELEVANT_INFO_FOUND'"
    retriever: Retriever = Field(exclude=True)
    
    def __init__(self, retriever: Retriever):
        super().__init__(retriever=retriever)
    
    def _run(self, query: str) -> str:
        try:
            results = self.retriever.retrieve(query)
            logger.info(f"Results From Retriever: {results}")
            
            if not results:
                return "NO_RELEVANT_INFO_FOUND"
            
            formatted_results = []
            for i, doc in enumerate(results, 1):
                formatted_result = f"[Source {i}]\n{doc.page_content}\n"
                if hasattr(doc.metadata, 'source'):
                    formatted_result += f"Source: {doc.metadata.source}\n"
                formatted_results.append(formatted_result)
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return "Error searching knowledge base"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class ConversationManager:
    """Manages conversations with the AI assistant"""
    
    def __init__(self):
        self.config = load_config()
        self._setup_ibm_clients()
        self._setup_models_and_embeddings()
        self._setup_vector_stores_and_retrievers()
        logger.info("Conversation manager initialized successfully")

    def _setup_ibm_clients(self):
        """Setup IBM API clients"""
        self.credentials = {
            "url": self.config["ibm_url"],
            "apikey": self.config["ibm_api_key"]
        }
        self.api_client = APIClient(self.credentials)
        self.api_client.set.default_project(self.config["ibm_project_id"])

    def _setup_models_and_embeddings(self):
        """Setup LLM and embedding models"""
        try:
            # Initialize crewai LLM
            self.llm = LLM(
                provider="ibm",
                model=self.config["model_id"],
                api_key=self.credentials["apikey"],
                project_id=self.config["ibm_project_id"],
                url=self.credentials["url"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
            
            # Setup embeddings
            self.embeddings = Embeddings(
                model_id=self.config["embedding_model_id"],
                credentials=self.credentials,
                project_id=self.config["ibm_project_id"]
            )
            logger.info("Models and embeddings setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up models and embeddings: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _setup_vector_stores_and_retrievers(self):
        """Setup vector stores and retrievers for member and employer knowledge bases"""
        try:
            # Employer VectorStore & Retriever
            self.employer_vector_store = VectorStore(
                api_client=self.api_client,
                connection_id=self.config["employer_connection_id"],
                collection_name="Employers_FAQ",
                embeddings=self.embeddings
            )
            self.employer_retriever = Retriever(
                vector_store=self.employer_vector_store,
                number_of_chunks=5
            )

            # Member VectorStore & Retriever
            self.member_vector_store = VectorStore(
                api_client=self.api_client,
                connection_id=self.config["member_connection_id"],
                collection_name="Members_FAQ",
                embeddings=self.embeddings
            )
            self.member_retriever = Retriever(
                vector_store=self.member_vector_store,
                number_of_chunks=5
            )
            logger.info("Vector stores and retrievers setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up vector stores and retrievers: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _create_agent(self, user_type: str) -> Agent:
        """Create an agent based on user type (member or employer)"""
        if not self.llm:
            raise ValueError("LLM not properly initialized")
            
        if user_type.lower() == "member":
            return Agent(
                role='CalPERS Member Assistant',
                goal='Help CalPERS members with their benefits, retirement planning, and member services',
                backstory="""You are a specialized CalPERS assistant focused on helping members. 
                You must ONLY use information retrieved from the knowledge base to answer questions.
                If the search tool returns 'NO_RELEVANT_INFO_FOUND', you must respond with 'I apologize, but I don't have enough information in my knowledge base to answer your question accurately.'
                Always cite your sources when providing information.""",
                llm=self.llm,
                verbose=True,
                tools=[SearchTool(self.member_retriever)],
                allow_delegation=False,
                max_iter=3
            )
        else:
            return Agent(
                role='CalPERS Employer Assistant',
                goal='Help CalPERS employers with their contributions, reporting requirements, and employer services',
                backstory="""You are a specialized CalPERS assistant focused on helping employers.
                You must ONLY use information retrieved from the knowledge base to answer questions.
                If the search tool returns 'NO_RELEVANT_INFO_FOUND', you must respond with 'I apologize, but I don't have enough information in my knowledge base to answer your question accurately.'
                Always cite your sources when providing information.""",
                llm=self.llm,
                verbose=True,
                tools=[SearchTool(self.employer_retriever)],
                allow_delegation=False,
                max_iter=3
            )

    def process_query(self, query: str, user_type: str) -> str:
        """Process a user query and return a response"""
        try:
            if not self.llm:
                raise ValueError("LLM not properly initialized")
                
            agent = self._create_agent(user_type)
            task = Task(
                description=f"""Help the user with their query: {query}
                Important Instructions:
                1. First, search the knowledge base using the search tool
                2. If no relevant information is found, respond with 'I apologize, but I don't have enough information in my knowledge base to answer your question accurately.'
                3. If information is found, use ONLY that information to answer the question
                4. Always include citations from the sources you used
                5. Do not make up or infer information that is not explicitly stated in the retrieved documents""",
                agent=agent,
                expected_output="A detailed and accurate response to the user's query based on the available information, with proper citations."
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=True)
            result = crew.kickoff()
            
            # Convert result to string
            response = str(result.raw_output) if hasattr(result, 'raw_output') else str(result)
            
            logger.info(f"Processed query for {user_type}: {query}")
            logger.info(f"Final Response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            return "I apologize, but I encountered an error processing your query. Please try again."
