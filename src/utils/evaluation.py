import pandas as pd
from typing import List, Dict, Optional
from loguru import logger

class ConversationTracker:
    def __init__(self):
        """Initialize the conversation tracker with an empty DataFrame"""
        self.df = pd.DataFrame(columns=['input', 'reference', 'output'])
        self.current_reference = None

    def update_reference(self, reference_text: str):
        """Update the current reference text from the knowledge base"""
        self.current_reference = reference_text

    def add_interaction(self, user_query: str, llm_response: str):
        """Add a new interaction to the DataFrame"""
        try:
            new_row = {
                'input': user_query,
                'reference': self.current_reference if self.current_reference else "NO_RELEVANT_INFO_FOUND",
                'output': llm_response
            }
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            logger.info(f"Added new interaction to evaluation DataFrame")
        except Exception as e:
            logger.error(f"Error adding interaction to DataFrame: {str(e)}")

    def get_evaluation_data(self) -> pd.DataFrame:
        """Return the current evaluation DataFrame"""
        return self.df

    def save_to_csv(self, filepath: str = "evaluation_data.csv"):
        """Save the evaluation data to a CSV file"""
        try:
            self.df.to_csv(filepath, index=False)
            logger.info(f"Saved evaluation data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving evaluation data: {str(e)}")

# Create a global instance
conversation_tracker = ConversationTracker() 