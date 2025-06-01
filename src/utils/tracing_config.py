from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

import os
from loguru import logger

from phoenix.otel import register

os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

tracer_provider = register(
  project_name="calpers-assistant",
  endpoint="https://app.phoenix.arize.com/v1/traces",
  auto_instrument=True
)

__all__ = ['setup_tracing']

def setup_tracing():
    """Setup OpenTelemetry tracing with Phoenix exporter"""
    try:
        # Check for required environment variables
        if not os.getenv("PHOENIX_API_KEY"):
            logger.warning("PHOENIX_API_KEY not found in environment variables. Tracing will be disabled.")
            return

        
        # Initialize CrewAI instrumentation
        CrewAIInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)
        
        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
        
        logger.info("Tracing setup completed successfully")
    except ImportError as e:
        logger.warning(f"Failed to import required tracing modules: {str(e)}. Tracing will be disabled.")
    except Exception as e:
        logger.error(f"Error setting up tracing: {str(e)}")
        logger.warning("Tracing will be disabled.") 