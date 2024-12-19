from flask import Flask, request, jsonify
from flask_cors import CORS
from .services.chat_services import ChatService


class ChatApp:
    """Main application class for the chat server."""
    
    def __init__(self):
        """Initialize the Flask application with CORS and chat service."""
        self.app = Flask(__name__)
        CORS(self.app)
        self.chat_service = ChatService()
        self._register_routes()

    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.route("/api/chat/<chat_id>", methods=['POST'])
        def chat(chat_id):
            """Handle incoming chat messages."""
            try:
                # Validate message format
                message = request.json
                if not message or "content" not in message:
                    print("Invalid message format")
                    return jsonify({"error": "Invalid message format"}), 400

                print(f"Received message: {message}")
                current_question = message["content"]

                # Process message and get response
                response = self.chat_service.process_message(
                    chat_id=chat_id,
                    question=current_question
                )

                # Create AI response message
                ai_message = {
                    "role": "assistant",
                    "content": response
                }

                return jsonify(ai_message)

            except Exception as e:
                print(f"Error in chat: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/chats")
        def get_all_chats():
            """Get all chat IDs."""
            try:
                return jsonify(self.chat_service.get_all_chats())
            except Exception as e:
                print(f"Error getting chats: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/chat/<chat_id>")
        def get_chat_messages(chat_id):
            """Retrieve all messages in a specific chat."""
            try:
                messages = self.chat_service.get_chat_messages(chat_id)
                return jsonify(messages)
            except Exception as e:
                print(f"Error getting chat messages: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/chat/<chat_id>", methods=['DELETE'])
        def delete_chat(chat_id):
            """Delete a chat by its ID."""
            try:
                if self.chat_service.delete_chat(chat_id):
                    return jsonify({"message": "Chat deleted successfully"})
                return jsonify({"error": "Chat not found"}), 404
            except Exception as e:
                print(f"Error deleting chat: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/chat/<chat_id>/rename", methods=['PUT'])
        def rename_chat(chat_id):
            """Rename a chat by its ID."""
            try:
                new_name = request.json.get('new_name')
                if not new_name:
                    return jsonify({"error": "New name is required"}), 400

                if self.chat_service.rename_chat(chat_id, new_name):
                    return jsonify({"message": "Chat renamed successfully"})
                return jsonify({"error": "Chat not found"}), 404
            except Exception as e:
                print(f"Error renaming chat: {str(e)}")
                return jsonify({"error": str(e)}), 500

    def run(self, debug=False):
        """Run the Flask application."""
        self.app.run(debug=debug)


def create_app():
    """Create and configure the application."""
    return ChatApp()


if __name__ == "__main__":
    app = create_app()
    app.run(debug=False)