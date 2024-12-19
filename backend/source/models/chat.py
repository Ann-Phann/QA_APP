from typing import Dict, Any, List

class ChatMessage:
    """ Represent a single message in a chat """

    def __init__(self, role: str, content: str):
        self.message = {
            "role": role,
            "content" : content
        }
        

    def to_dict(self) -> Dict[str, str]:
        return self.message
    
    @property
    def role(self) ->str:
        return self.message["role"]
    
    @property
    def content(self) -> str:
        return self.message["content"]
    
class ChatHistory:
    """ Represent the history of a chat and context."""

    def __init__(self):
        self.chats: Dict[str, List[ChatMessage]] = {}

    def add_message(self, chat_id: str, message: ChatMessage):
        if chat_id not in self.chats:
            self.chats[chat_id] = []
        self.chats[chat_id].append(message)

    def get_chat(self, chat_id: str) -> List[ChatMessage]:
        return self.chats.get(chat_id, [])
    
    def delete_chat(self,chat_id: str) -> bool:
        if chat_id in self.chats:
            del self.chats[chat_id]
            return True
        return False
    
    def rename_chat(self, old_id: str, new_if: str) -> bool:
        if old_id in self.chats:
            self.chats[new_if] = self.chats.pop(old_id)
            return True
        return False
        