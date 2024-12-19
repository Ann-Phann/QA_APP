from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.prompts import ChatPromptTemplate
#from langchain.llms import OllamaLLM
from langchain_ollama import OllamaLLM

# Define template
template = """
Answer the question based on the context provided.
Context: {context}
Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model='llama3.1')
chain = prompt | model

app = Flask(__name__)
CORS(app)  # Allow React app to access our API

# Store chats in a simple dictionary
chat_history = {}

@app.route("/api/chat/<chat_id>", methods=['POST'])
def chat(chat_id):
    message = request.json
    
    # Create a new chat list if this chat_id is new
    if chat_id not in chat_history:
        chat_history[chat_id] = []
    
    response = chain.invoke({"context": chat_history[chat_id], "question": message['content']})
    
    # Save the user's message
    chat_history[chat_id].append({
        "role": "user",
        "content": message['content']
    })
    
    # Get AI response
    # response = ollama.chat(
    #     model='llama3.1',
    #     messages=chat_history[chat_id]
    # )
    
    # Save and return AI's response
    ai_message = {
        "role": "assistant",
        #"content": response['message']['content']
        "content": response
    }
    chat_history[chat_id].append(ai_message)
    
    return jsonify(ai_message)

@app.route("/api/chats")
def get_all_chats():
    return jsonify(list(chat_history.keys()))

@app.route("/api/chat/<chat_id>")
def get_chat_messages(chat_id):
    if chat_id not in chat_history:
        return jsonify([])
    return jsonify(chat_history[chat_id])

@app.route("/api/chat/<chat_id>", methods=['DELETE'])
def delete_chat(chat_id):
    if chat_id in chat_history:
        del chat_history[chat_id]
        return jsonify({"message": "Chat deleted successfully"})
    return jsonify({"error": "Chat not found"}), 404

@app.route("/api/chat/<chat_id>/rename", methods=['PUT'])
def rename_chat(chat_id):
    new_name = request.json.get('new_name')
    
    if not new_name:
        return jsonify({"error": "New name is required"}), 400
        
    if chat_id in chat_history:
        # Store the chat content
        chat_content = chat_history[chat_id]
        # Delete old chat
        del chat_history[chat_id]
        # Create new chat with new name
        chat_history[new_name] = chat_content
        return jsonify({"message": "Chat renamed successfully"})
    
    return jsonify({"error": "Chat not found"}), 404

if __name__ == "__main__":
    app.run(debug=True) 

