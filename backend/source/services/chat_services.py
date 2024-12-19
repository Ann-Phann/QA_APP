from typing import List, Dict
from ..models.chat import ChatHistory, ChatMessage
from ..agents.specialized_agent import QuestionClassificationAgent, RelevancyAgent, ResponseAgent, SummaryAgent
from ..utils.templates import OutputFormatter

class ChatService:
    """ Handlechat operations and agent selection """

    def __init__(self):
        self.chat_history = ChatHistory()
        self.classifier_agent = QuestionClassificationAgent("Classifier")
        self.relevancy_agent = RelevancyAgent("Relevancy")
        self.response_agent = ResponseAgent("Response")
        self.summary_agent = SummaryAgent("Summary")
        self.output_formatter = OutputFormatter()

    def process_message(self, chat_id: str, question: str) -> str:
        try:
                
            # create new chat list if chat_id is new
            if chat_id not in self.chat_history.chats:
                self.chat_history.chats[chat_id] = []

            # Get context and check relevancy bewtween current question and previous
            context = self.get_relevant_context(chat_id, question)
            print(f"Context: {context}")

            # Default classification
            question_type = "Other"
            reasoning_examples = "Provide step-by-step reasoning based on the question"
            # Classify the question type
            try:
                classification_response = self.classifier_agent.process(question= question)
                print(f"Question type: {classification_response}")

                if isinstance(classification_response,str):
                    # if a string, assume that is question type
                    question_type = classification_response
                else:
                    # if it's a dictionary, extract the type and reasoning
                    question_type = classification_response.get("type", "Other")
            
            except Exception as e:
                print(f"Error in classifying question type: {e}")


            # Get output format and reasoning examples
            output_format = self.output_formatter.get_output_format(question_type)
            reasoning_examples = self.output_formatter.get_reasoning_examples(question_type)

            # Generate response
            try:
                response = self.response_agent.process(
                    context = context,
                    question = question,
                    reasoning_examples = reasoning_examples,
                    output_format = output_format
                )
                print(f"Got response: {response}")

                # Enforce to say Yes/No for Yes/No question type
                if question_type == "Yes/No":
                    response = self.response_agent.enforce_yes_no_format(
                        response,
                        question,
                        context,
                        reasoning_examples,
                        output_format
                    )

                # Store the chat history
                self.chat_history.add_message(chat_id, ChatMessage("user", question))
                self.chat_history.add_message(chat_id, ChatMessage("assistant", response))

                return response
            
            except Exception as e:
                print(f"Error in generating response: {e}")
                raise

        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            raise
    
    """ get relevant context by checking relevancy between previous and current question """
    def get_relevant_context(self, chat_id: str, current_question:str) -> str:
        messages = self.chat_history.get_chat(chat_id)

        if not messages:
            return ""
        
        # If this is the first question, store as current topic
        if len(messages) == 0:
            return ""
        
        # Get the previous question
        prev_question = messages[-1].content

        # Check if the questions are related
        relevancy = self.relevancy_agent.process(
            prev_question = prev_question,
            current_question = current_question
        )

        # Get comprehensive summary include both previous and new/current context
        # if "related" in relevancy.lower():
        #     summarized_history = self.summarize_chat_history(messages, current_question, True)
        #     return f"Relevant chat history (Related topic): {summarized_history}"
        # else:
        #     return f"New topic: {summarized_history}."

        is_related = "related" in relevancy.lower()

        summarized_history = self.summarize_chat_history(
            messages,
            current_question,
            is_related
        )
        return summarized_history
        
    def summarize_chat_history(self, messages: List[ChatMessage], current_question: str, is_related: bool) -> str:
        """
        Stragegy:
        - If questions are related: include previous context + current question.
        - If starting new topic: mark as new topic but still keep previous information.
        - Track multiple topics in conversation.
        """

        history_text = "\n".join({
            f"{msg.role}: {msg.content}"
            for msg in messages
        })

        # Call summary agent for enhanced summary.
        return self.summary_agent.process(
            chat_history = history_text,
            question = current_question,
            is_related = str(is_related)
        )
    
    def get_all_chats(self) ->List[str]:
        """ Get all chat_id"""
        return list(self.chat_history.chats.keys())
    
    def get_chat_messages(self, chat_id:str) -> List[Dict]:
        """ Get all messages for a specific tasks"""
        messages = self.chat_history.get_chat(chat_id)
        return [msg.to_dict() for msg in messages]
    
    def delete_chat(self, chat_id:str) -> bool:
        return self.chat_history.delete_chat(chat_id)
    
    def rename_chat(self, old_id: str, new_id: str) -> bool:
        return self.chat_history.rename_chat(old_id, new_id)
    

        
            