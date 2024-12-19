from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from abc import ABC, abstractmethod

"""
Abstract base class for all AI agents in the system.
"""
class BaseAgent:

    def __init__(self, name:str, model_name:str = "llama3.2"):
        self.name = name
        self.model = OllamaLLM(model=model_name)
        self.prompt_tepmplate = self.get_prompt_template()
        self.chain = ChatPromptTemplate.from_template(self.prompt_tepmplate) | self.model

    @abstractmethod
    def get_prompt_template(self) ->str:
        """ Returns the prompt template for the agent """
        pass

    @abstractmethod
    def process(self, **kwargs):
        """ Processes the input and returns the response """
        pass