from phoenix.evals import (
    OpenAIModel,
    llm_classify,
    RAG_RELEVANCY_PROMPT_TEMPLATE,
    RAG_RELEVANCY_PROMPT_RAILS_MAP,
    HALLUCINATION_PROMPT_TEMPLATE,
    HALLUCINATION_PROMPT_RAILS_MAP,
    TOXICITY_PROMPT_TEMPLATE,
    TOXICITY_PROMPT_RAILS_MAP
)
from loguru import logger
import pandas as pd

class RAGEvaluator:
    def __init__(self):
        """Initialize the RAG evaluator with OpenAI model"""
        self.model = OpenAIModel(
            model_name="gpt-4",
            temperature=0.0,
        )
        self.relevance_rails = list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values())
        self.hallucination_rails = list(HALLUCINATION_PROMPT_RAILS_MAP.values())
        self.toxicity_rails = list(TOXICITY_PROMPT_RAILS_MAP.values())

    def evaluate_relevance(self, df: pd.DataFrame) -> dict:
        """
        Evaluate the relevance, hallucination, and toxicity of RAG responses using Phoenix
        
        Args:
            df: DataFrame containing 'input', 'reference', and 'output' columns
            
        Returns:
            dict: Raw classification results
        """
        try:
            # Evaluate relevance
            relevance_classifications = llm_classify(
                dataframe=df,
                template=RAG_RELEVANCY_PROMPT_TEMPLATE,
                model=self.model,
                rails=self.relevance_rails,
                provide_explanation=True
            )
            
            # Evaluate hallucination
            hallucination_classifications = llm_classify(
                dataframe=df,
                template=HALLUCINATION_PROMPT_TEMPLATE,
                model=self.model,
                rails=self.hallucination_rails,
                provide_explanation=True
            )
            
            # Evaluate toxicity
            toxicity_classifications = llm_classify(
                dataframe=df,
                template=TOXICITY_PROMPT_TEMPLATE,
                model=self.model,
                rails=self.toxicity_rails,
                provide_explanation=True
            )
            
            # Create DataFrames for each classification type
            relevance_df = relevance_classifications            
            hallucination_df = hallucination_classifications
            
            toxicity_df =  toxicity_classifications
            
            return {
                "classifications": {
                    "relevance": relevance_df,
                    "hallucination": hallucination_df,
                    "toxicity": toxicity_df
                }
            }
        except Exception as e:
            logger.error(f"Error in RAG evaluation: {str(e)}")
            return {
                "classifications": {
                    "relevance": pd.DataFrame(),
                    "hallucination": pd.DataFrame(),
                    "toxicity": pd.DataFrame()
                }
            }

# Create a global instance
rag_evaluator = RAGEvaluator()