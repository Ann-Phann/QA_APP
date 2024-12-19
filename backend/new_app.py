from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Store chats in a simple dictionary
chat_history = {}

# Initialize LLM model
model = OllamaLLM(model='llama3.2')

# Single versatile prompt template
TEMPLATE = """
Context: {context}

User Question: {question}

Reasoning examples: {reasoning_examples}

Output Format: {output_format}

Please provide a response following the output format.
Your response:
"""
prompt = ChatPromptTemplate.from_template(TEMPLATE)
chain = prompt | model

# Chat history summarize template
summary_template = """
Please summarize the following context into a brief, focused statement that captures the main intent:

Chat History: {chat_history}

Current Question: {question}

Provide a clear, concise summary in 1-2 sentences:
"""

summary_prompt = ChatPromptTemplate.from_template(summary_template)
summary_chain = summary_prompt | model  

# Define a classification template for types of question
classification_question_template = """
Classify the following question into one of the type: Yes/No, Explanation, List, Comparison, or Other.

Question: {question}

Instruction:
- Yes/No questions typically start with words like "is", "are", "can", "do", "will", etc.
- Explanation questions ofter start with "why", "how", "explain", etc.
- List questions usually contain words like "list", "steps", "point out", "what are the stages", "how to", etc.
- Comparision questions often include "difference", "compare", "distinguish", etc, and involve comparing objects/etc.
- If the question does not fit into any of these categories, classify it as "Other".

Provide the type of the question and a brief reasoning for your classification: 
"""
classification_prompt = ChatPromptTemplate.from_template(classification_question_template)
classification_chain = classification_prompt | model


# Add new template for relevancy checking
relevancy_template = """
Determine if these two questions are related topics (answer only with "related" or "unrelated"):

Previous Question: {prev_question}
Current Question: {current_question}

Consider them related if they:
- Discuss the same topic
- Build upon previous context
- Reference similar concepts
"""

relevancy_prompt = ChatPromptTemplate.from_template(relevancy_template)
relevancy_chain = relevancy_prompt | model



# Summarize chat history for context
def summarize_chat_history(chat_history, current_question):
    # Format chat history for summarization
    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in chat_history
    ])
    return summary_chain.invoke({"chat_history": history_text, "question": current_question})

def get_output_format(question_type):
    if question_type == "Yes/No":
        return "Start with a clear 'Yes' or 'No' followed by a brief explanation."
    elif question_type == "Explanation":
        return "Provide a clear and concise explanation."
    elif question_type == "List":
        return "Provide your response as a numbered list of points."
    elif question_type == "Comparison":
        return "Structure your response with clear comparisons."
    else:
        return "Provide a clear, concise response."
    
def get_reasoning_examples(question_type):
    if question_type == "Yes/No":
        return [
            {"question": "Is water wet?", "response": "Yes. Water is wet because it adheres to surfaces and creates the sensation of wetness."},
            {"question": "Is fire cold?", "response": "No. Fire produces heat and is not cold."}
        ]
    elif question_type == "Explanation":
        return [{"question": "Why is the sky blue?", "response": "The sky appears blue due to the scattering of sunlight by the Earth's atmosphere."}]
    elif question_type == "List":
        return [
            {"question": "What are the primary colors?", "response": "The primary colors are red, blue, and yellow."},
            {"question": "List the planets in the solar system.", "response": "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune."}
        ]
    elif question_type == "Comparison":
        return [
            {"question": "What is the difference between a lion and a tiger?", "response": "Lions are social and live in groups called prides, while tigers are solitary and have striped coats."},
            {"question": "Compare a desktop computer and a laptop.", "response": "A desktop computer is stationary and more powerful, while a laptop is portable and compact."}
        ]
#(response)
def enforce_yes_no_format(response, current_question, context, reasoning_examples, output_format):

    # Check if the response starts with "Yes" or "No"
    if response.lower().startswith("yes") or response.lower().startswith("no"):
        return response  # Already in the correct format

    # If not, regenerate the response
    print(f"Response format is incorrect. Regenerating response following {output_format}")
    response = chain.invoke({
        "context": context,
        "query": current_question,
        "reasoning_examples": reasoning_examples,
        "output_format": "Respond with 'Yes' or 'No' followed by a brief explanation."
    })

    return response

@app.route("/api/chat/<chat_id>", methods=['POST'])
def chat(chat_id):
    try:
        message = request.json  # User's query message
        if not message or "content" not in message:
            print("Invalid message format")
            return jsonify({"error": "Invalid message format"}), 400

        print(f"Received message: {message}")
        current_question = message["content"]

        # Create a new chat list if this chat_id is new
        if chat_id not in chat_history:
            chat_history[chat_id] = []

        # Check relevancy with previous question if exists
        context = ""
        if chat_history[chat_id]:
            prev_question = chat_history[chat_id][-1]["content"]
            relevancy = relevancy_chain.invoke({
                "prev_question": prev_question,
                "current_question": current_question
            })
            
            # Only include history summary if questions are related
            if "related" in relevancy.lower():
                summarized_history = summarize_chat_history(chat_history[chat_id], current_question)
                context = f"Relevant Chat History:\n{summarized_history}"
            else:
                context = "No relevant previous context."


        # Note: mention about questuin type, different agent, fine-tunning

        # Default classification
        question_type = "Other"
        reasoning_examples = "Provide step-by-step reasoning based on the question type."

        # Classify the question type using AI
        # Add reasoning examples (optional, for CoT)
        try: 
            classification_response = classification_chain.invoke({
                "question" : current_question
            })
            print(f"Question classification response {classification_response}")

            # Extract the type
            # question_type = classification_response.get("type", "Other") 
            # reasoning_examples = classification_response.get("reasoning", "")
            if isinstance(classification_response, str):
                # If it's a string, we can assume it's the classification type
                question_type = classification_response
                # reasoning_examples = "No specific reasoning provided."
            else:
                # If it's a dictionary, extract the type and reasoning
                question_type = classification_response.get("type", "Other")
                # reasoning_examples = classification_response.get("reasoning", "")
        except Exception as e:
            print(f"Error in classifying question type: {e}")


        reasoning_examples = get_reasoning_examples(question_type)

        output_format = get_output_format(question_type)

        

        # Send prompt to LLM and get the response
        try:
            response = chain.invoke({
                "context": context, 
                "question": current_question,
                "reasoning_examples": reasoning_examples,
                "output_format": output_format
            })
            print(f"LLM response received: {response}")

            # Enforce Yes/No question format 
            if question_type == "Yes/No":
                response = enforce_yes_no_format(response, current_question, context,reasoning_examples, output_format)

        except Exception as e:
            print(f"Error in getting LLM response: {e}")
            return jsonify({"error": f"LLM error: {str(e)}"}), 500
            
        # Add user's query to chat history
        chat_history[chat_id].append({
            "role": "user", 
            "content": current_question
        })
        # chat_history[chat_id].append({"role": "user", "content":message["content"]})

        # Save AI response
        ai_message = {
            "role": "assistant",
            "content": response
        }
        # Add AI response to chat history
        chat_history[chat_id].append(ai_message)

        # Return AI response
        return jsonify(ai_message)
    
    except Exception as e:
        print(f"General: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chats")
def get_all_chats():
    """Get all chat IDs."""
    return jsonify(list(chat_history.keys()))

@app.route("/api/chat/<chat_id>")
def get_chat_messages(chat_id):
    """Retrieve all messages in a specific chat."""
    if chat_id not in chat_history:
        return jsonify([])
    return jsonify(chat_history[chat_id])

@app.route("/api/chat/<chat_id>", methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat by its ID."""
    if chat_id in chat_history:
        del chat_history[chat_id]
        return jsonify({"message": "Chat deleted successfully"})
    return jsonify({"error": "Chat not found"}), 404

@app.route("/api/chat/<chat_id>/rename", methods=['PUT'])
def rename_chat(chat_id):
    """Rename a chat by its ID."""
    new_name = request.json.get('new_name')
    if not new_name:
        return jsonify({"error": "New name is required"}), 400

    if chat_id in chat_history:
        # Store the chat content and rename the chat
        chat_content = chat_history[chat_id]
        del chat_history[chat_id]
        chat_history[new_name] = chat_content
        return jsonify({"message": "Chat renamed successfully"})

    return jsonify({"error": "Chat not found"}), 404

if __name__ == "__main__":
    app.run(debug=False)
