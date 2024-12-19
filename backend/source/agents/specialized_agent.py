from .base_agent import BaseAgent

class QuestionClassificationAgent(BaseAgent):
    """ Agent specialized in classifying questions into different types """
    def get_prompt_template(self) ->str:
        return """
        Classify the following question into one of the type: Yes/No, Explanation, List, Comparison, or Other.

        Question: {question}

        Instruction:
        - Yes/No questions typically start with words like "is", "are", "can", "do", "will", etc.
        - Explanation questions ofter start with "why", "how", "explain", etc.
        - List questions usually contain words like "list", "steps", "point out", "what are the stages", "how to", etc.
        - Comparision questions often include "difference", "compare", "distinguish", etc, and involve comparing objects/etc.
        - If the question does not fit into any of these categories, classify it as "Other".

        Provide the type of the question and a brief reasoning for your classification: Question Classification        
        """

    def process(self, **kwargs):
        return self.chain.invoke(kwargs)
    
class RelevancyAgent(BaseAgent):
    """ Agent specialized in determining the relevancy of a question """
    def get_prompt_template(self) ->str:
        return """
        Determine if these two questions are related topics (answer only with "related" or "unrelated"):

        Previous Question: {prev_question}
        Current Question: {current_question}

        Consider them related if they:
        - Discuss the same topic
        - Build upon previous context
        - Reference similar concepts
        """

    def process(self, **kwargs):
        return self.chain.invoke(kwargs)
    
class ResponseAgent(BaseAgent):
    """ Main in generating responses to questions """
    def get_prompt_template(self) ->str:
        return """
        Context: {context}

        User Question: {question}

        Reasoning examples: {reasoning_examples}

        Output Format: {output_format}

        Please provide a response following the output format.
        """

    def process(self, **kwargs):
        try:
            response = self.chain.invoke(kwargs)

            # If it is Yes/No question
            if kwargs.get("output_format", "").startswith("Start with a clear 'Yes' or 'No'"):
                response = self.enforce_yes_no_format(
                    response,
                    kwargs["question"],
                    kwargs["context"],
                    kwargs["reasoning_examples"],
                    kwargs["output_format"]
                )

            return response
        except Exception as e:
            print(f"Error in ResponseAgent process: {str(e)}")
            raise
    
    def enforce_yes_no_format(self,response, current_Question, context, reasoning_examples, output_format) -> str:
        """ Enforce the format to Yes/No type questions. """

        # Check if start with "Yes" or "No"
        if response.lower().startswith("yes") or response.lower().startswith("no"):
            return response #in correct form
        
        # if not, regenerate response
        print(f"Response format is incorrect for Yes/No question. Regenerate following format {output_format}")
        response = self.chain.invoke({
            "context": context,
            "question": current_Question,
            "reasoning_examples" : reasoning_examples,
            "output_format": "Response with 'Yes' or 'No' followed by a brief explanation"
        })

        return response

        
    
class SummaryAgent(BaseAgent):
    """ Agent specialized in summarizing the chat history """

    def get_prompt_template(self) ->str:
        return """
        Please analyze and summarize the conversation context:

        Chat History: {chat_history}

        Current Question: {question}

        Is related to previous topic: {is_related}

        Instructions:
        1. For related topics:
        - Maintain continuity with previous context.
        - Highlight relevant information.
        - Show connections between points.

        2. For new topics:
        - Mark transition to new topic.
        - Keep relevant background information.
        - Note any connections to previous topics.

        3. Always include:
        - Key facts and conclusion.
        - Important context.
        - Relevant background.

        Summary:
        """

    def process(self, **kwargs):
        try:
            summary = self.chain.invoke(kwargs)
            print(f"Generated summary: {summary}")
            return summary
        except Exception as e:
            print(f"Error in summary generation: {e}")
            raise

