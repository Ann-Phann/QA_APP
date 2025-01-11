from typing import List, Dict
from ..models.chat import ChatHistory, ChatMessage
from ..agents.specialized_agent import QuestionClassificationAgent, RelevancyAgent, ResponseAgent, SummaryAgent
from ..utils.templates import OutputFormatter
from ..services.document_processor import DocumentProcessor
from langchain_core.documents import Document
import chromadb
from transformers import pipeline
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class ChatService:
    """ Handlechat operations and agent selection """

    def __init__(self):
        self.chat_history = ChatHistory()
        self.classifier_agent = QuestionClassificationAgent("Classifier")
        self.relevancy_agent = RelevancyAgent("Relevancy")
        self.response_agent = ResponseAgent("Response")
        self.summary_agent = SummaryAgent("Summary")
        self.output_formatter = OutputFormatter()
        self.document_processor = DocumentProcessor()
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_collection("documents")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # # Load documents into the collection
        # documents = self.document_processor.load_documents()
        # print(f"load documents function called")
        # for doc in documents:
        #     if isinstance(doc, Document):
        #     # Split the document into chunks
        #         text = doc.page_content
        #         chunks = self.document_processor.text_splitter.split_text(text)
        #     else:
        #         raise TypeError(f"Expected a Document object, got {type(doc)}")
        #     # Generate a unique identifier for the document
        #     ids = [str(uuid4()) for _ in chunks] # Create a list of UUIDs as IDs

        #     # Embed the chunks
        #     embeddings = self.document_processor.embed_documents(chunks)
            
        #     # Add the embeddings and chunks to the ChromaDB collection
        #     self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)

        # Process and embed documents using DocumentProcessor
        self.document_processor.process_documents()
        print("Documents processed and embedded.")
       
    def process_message(self, chat_id: str, question: str) -> str:
        try:
            # check if the question require document search RAG
            if self.is_factual_question(question):
                return self.retrieve_from_rag(question)
            
            #existing logic for general question
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

                print(f"After RAG")
                return response
            
            except Exception as e:
                print(f"Error in generating response: {e}")
                raise

        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            raise
    
    # use zero-shot classification to determine if the question is factual
    def is_factual_question(self, question: str) -> bool:
        """Determine if the question is factual and requires specific information."""
        # Categories of question
        candidate_labels = ["factual", "general"]
        result = self.classifier(question, candidate_labels)

        # check if the highest score corresponds to "factual"
        return result['labels'][0] == "factual"

    def retrieve_from_rag(self, question: str) -> str:
        """Retrieve relevant information from Chroma for factual questions."""
        try:
            question_embedding = self.document_processor.embed_documents([question])
            results = self.collection.query(
                query_embeddings=question_embedding, 
                n_results=10  # Get more results for ranking
            )
            
            if not results['documents']:
                return "No relevant information found."
            
            # Rank chunks based on semantic similarity to question
            ranked_chunks = self._rank_chunks(question, results['documents'][0])
            
            # Get top chunks
            top_chunks = ranked_chunks[:3]
            
            # Create a summarization prompt
            summary_prompt = f"""
            Question: {question}
            
            Context from documents:
            {' '.join(chunk for chunk, _ in top_chunks)}
            
            Please provide a clear and concise answer based on the above context. 
            Focus only on information that directly answers the question.
            If specific numbers, dates, or criteria are mentioned, include those.
            """
            
            # Get summarized answer from response agent
            summarized_answer = self.response_agent.process(
                context="",  # Context is already in the prompt
                question=summary_prompt,
                reasoning_examples="Provide direct, factual information from the context.",
                output_format="Concise answer with bullet points if multiple items exist."
            )
            
            return summarized_answer
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {e}")
            return "Error occurred while retrieving information"
        

    def extract_relevant_info(self, document: str, question: str) -> str:
        """Extract or summarize relevant information from a document."""
        # Tokenize the document and question
        document_sentences = self.tokenize_document(document)
        question_tokens = self.tokenize_question(question)

        # Find the most relevant sentence
        best_sentence = None
        highest_similarity = 0

        for sentence in document_sentences:
            similarity = self.calculate_similarity(sentence, question_tokens)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_sentence = sentence

        return best_sentence if best_sentence else "No specific information found."
    
    def tokenize_document(self, document: str) -> List[str]:
        """Tokenize the document into sentences using RecursiveCharacterTextSplitter."""
        # Assuming self.document_processor.text_splitter is an instance of RecursiveCharacterTextSplitter
        return self.document_processor.text_splitter.split_text(document)

    def tokenize_question(self, question: str) -> List[str]:
        """Tokenize the question."""
        # Implement a method to tokenize the question
        return question.split()

    def calculate_similarity(self, sentence: str, question_tokens: List[str]) -> float:
        """Calculate similarity between a sentence and the question."""
        # Implement a method to calculate similarity, e.g., using cosine similarity
        # Placeholder: return a mock similarity score
        return 0.5  # Replace with actual similarity calculation

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

    def _rank_chunks(self, question: str, chunks: List[str]) -> List[tuple]:
        """Rank chunks based on semantic similarity to question."""
        try:
            # Get embeddings for question and chunks
            question_embedding = self.document_processor.embed_documents([question])[0]
            chunk_embeddings = self.document_processor.embed_documents(chunks)
            
            # Calculate similarity scores
            ranked_results = []
            for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
                # Calculate cosine similarity
                similarity = self._calculate_similarity(question_embedding, chunk_embedding)
                ranked_results.append((chunk, similarity))
            
            # Sort by similarity score in descending order
            return sorted(ranked_results, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in ranking chunks: {e}")
            return [(chunk, 0.0) for chunk in chunks]

    def _calculate_similarity(self, embedding1, embedding2) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize the embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0