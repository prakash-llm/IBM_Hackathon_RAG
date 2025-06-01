from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
from ibm_watsonx_ai.foundation_models.extensions.rag import VectorStore, Retriever
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from loguru import logger
from pydantic import Field
import traceback
from opentelemetry import trace

from src.utils.config import load_config
from src.utils.evaluation import conversation_tracker
from src.utils.rag_evaluation import rag_evaluator

from phoenix.evals import (
    OpenAIModel)
model = OpenAIModel(
    model_name="gpt-4",
    temperature=0.0,
)

# Get tracer
tracer = trace.get_tracer(__name__)

class SearchTool(BaseTool):
    """Tool for searching the knowledge base"""
    name: str = "search_knowledge_base"
    description: str = "Search the knowledge base for relevant information. If no relevant information is found, return 'NO_RELEVANT_INFO_FOUND'"
    retriever: Retriever = Field(exclude=True)
    
    def __init__(self, retriever: Retriever):
        super().__init__(retriever=retriever)
    
    def _run(self, query: str) -> str:
        with tracer.start_as_current_span("search_knowledge_base") as span:
            try:
                span.set_attribute("query", query)
                results = self.retriever.retrieve(query)
                logger.info(f"Results From Retriever: {results}")
                
                if not results:
                    span.set_attribute("results_found", False)
                    conversation_tracker.update_reference("NO_RELEVANT_INFO_FOUND")
                    return "NO_RELEVANT_INFO_FOUND"
                
                span.set_attribute("results_found", True)
                span.set_attribute("num_results", len(results))
                
                formatted_results = []
                reference_text = []
                for i, doc in enumerate(results, 1):
                    formatted_result = f"[Source {i}]\n{doc.page_content}\n"
                    if hasattr(doc.metadata, 'source'):
                        formatted_result += f"Source: {doc.metadata.source}\n"
                    formatted_results.append(formatted_result)
                    reference_text.append(doc.page_content)
                
                # Update the reference text for evaluation
                conversation_tracker.update_reference("\n".join(reference_text))
                
                return "\n\n".join(formatted_results)
            except Exception as e:
                span.set_attribute("error", str(e))
                logger.error(f"Error searching knowledge base: {str(e)}")
                logger.error(traceback.format_exc())
                return "Error searching knowledge base"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class ConversationManager:
    """Manages conversations with the AI assistant"""
    
    def __init__(self):
        with tracer.start_as_current_span("initialize_conversation_manager") as span:
            try:
                self.config = load_config()
                self._setup_ibm_clients()
                self._setup_models_and_embeddings()
                self._setup_vector_stores_and_retrievers()
                logger.info("Conversation manager initialized successfully")
                span.set_attribute("status", "success")
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error", str(e))
                raise

    def _setup_ibm_clients(self):
        """Setup IBM API clients"""
        with tracer.start_as_current_span("setup_ibm_clients") as span:
            try:
                self.credentials = {
                    "url": self.config["ibm_url"],
                    "apikey": self.config["ibm_api_key"]
                }
                self.api_client = APIClient(self.credentials)
                self.api_client.set.default_project(self.config["ibm_project_id"])
                span.set_attribute("status", "success")
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error", str(e))
                raise

    def _setup_models_and_embeddings(self):
        """Setup LLM and embedding models"""
        with tracer.start_as_current_span("setup_models_and_embeddings") as span:
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
                span.set_attribute("status", "success")
                span.set_attribute("model_id", self.config["model_id"])
                span.set_attribute("embedding_model_id", self.config["embedding_model_id"])
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error", str(e))
                logger.error(f"Error setting up models and embeddings: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def _setup_vector_stores_and_retrievers(self):
        """Setup vector stores and retrievers for member and employer knowledge bases"""
        with tracer.start_as_current_span("setup_vector_stores") as span:
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
                span.set_attribute("status", "success")
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error", str(e))
                logger.error(f"Error setting up vector stores and retrievers: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def _create_agent(self, user_type: str) -> Agent:
        """Create an agent based on user type (member or employer)"""
        with tracer.start_as_current_span("create_agent") as span:
            try:
                if not self.llm:
                    raise ValueError("LLM not properly initialized")
                
                span.set_attribute("user_type", user_type)
                
                if user_type.lower() == "member":
                    agent = Agent(
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
                    agent = Agent(
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
                span.set_attribute("status", "success")
                return agent
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error", str(e))
                raise

    def process_query(self, query: str, user_type: str) -> tuple[str, dict]:
        """Process a user query and return a response and evaluation results"""
        with tracer.start_as_current_span("process_query") as span:
            try:
                span.set_attribute("query", query)
                span.set_attribute("user_type", user_type)
                
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
                
                # Add the interaction to the evaluation tracker
                conversation_tracker.add_interaction(query, response)
                
                logger.info(f"Processed query for {user_type}: {query}")
                logger.info(f"Final Response: {response}")
                
                span.set_attribute("status", "success")
                span.set_attribute("response_length", len(response))

                # Get evaluation results
                df = conversation_tracker.get_evaluation_data()
                eval_results = rag_evaluator.evaluate_relevance(df)
                
                return response, eval_results
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error", str(e))
                logger.error(f"Error processing query: {str(e)}")
                logger.error(traceback.format_exc())
                return "I apologize, but I encountered an error processing your query. Please try again.", {
                    "classifications": [],
                    "metrics": {
                        "total_queries": 0,
                        "relevant_count": 0,
                        "relevance_score": 0
                    }
                }
