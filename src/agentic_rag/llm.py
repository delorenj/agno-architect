from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from .interfaces import LanguageModelProvider
from .config import LanguageModelConfig

class OpenAILanguageModel(LanguageModelProvider):
    """Concrete implementation of LanguageModelProvider for OpenAI models."""
    
    def __init__(self, config: LanguageModelConfig):
        self.config = config

    def get_agent_model(self) -> BaseLanguageModel:
        """Get a language model configured for agent tasks."""
        return ChatOpenAI(
            model=self.config.agent_model,
            temperature=self.config.temperature,
            streaming=True
        )

    def get_grader_model(self) -> BaseLanguageModel:
        """Get a language model configured for grading tasks."""
        return ChatOpenAI(
            model=self.config.grader_model,
            temperature=self.config.temperature
        )

    def get_generator_model(self) -> BaseLanguageModel:
        """Get a language model configured for generation tasks."""
        return ChatOpenAI(
            model=self.config.generator_model,
            temperature=self.config.temperature,
            streaming=True
        )

def create_language_model_provider(config: LanguageModelConfig) -> LanguageModelProvider:
    """Factory function for creating language model providers."""
    return OpenAILanguageModel(config)