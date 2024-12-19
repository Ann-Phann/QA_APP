import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [chats, setChats] = useState([]);
  const [currentChat, setCurrentChat] = useState(null);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    fetchChats();
  }, []);

  const fetchChats = async () => {
    const response = await fetch('http://localhost:5000/api/chats');
    const data = await response.json();
    setChats(data);
  };

  const createNewChat = () => {
    const chatId = `chat_${Date.now()}`;
    setChats([...chats, chatId]);
    setCurrentChat(chatId);
    setMessages([]);
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || !currentChat) return;

    const newMessage = { role: 'user', content: input };
    setMessages([...messages, newMessage]);
    setInput('');

    try {
      const response = await fetch(`http://localhost:5000/api/chat/${currentChat}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newMessage),
      });
      const data = await response.json();
      setMessages([...messages, newMessage, data]);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const loadChat = async (chatId) => {
    setCurrentChat(chatId);
    const response = await fetch(`http://localhost:5000/api/chat/${chatId}`);
    const data = await response.json();
    setMessages(data);
  };

  const deleteChat = async (e, chatId) => {
    e.stopPropagation();
    try {
      await fetch(`http://localhost:5000/api/chat/${chatId}`, {
        method: 'DELETE',
      });
      setChats(chats.filter(id => id !== chatId));
      if (currentChat === chatId) {
        setCurrentChat(null);
        setMessages([]);
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
    }
  };

  const renameChat = async (e, chatId) => {
    e.stopPropagation();
    const newName = prompt('Enter new name for chat:');
    if (!newName) return;

    try {
      await fetch(`http://localhost:5000/api/chat/${chatId}/rename`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ new_name: newName }),
      });
      setChats(chats.map(id => id === chatId ? newName : id));
      if (currentChat === chatId) {
        setCurrentChat(newName);
      }
    } catch (error) {
      console.error('Error renaming chat:', error);
    }
  };

  return (
    <div className="App">
      <div className="sidebar">
        <button onClick={createNewChat}>New Chat</button>
        <div className="chat-list">
          {chats.map((chatId) => (
            <div
              key={chatId}
              className={`chat-item ${currentChat === chatId ? 'active' : ''}`}
            >
              <span onClick={() => loadChat(chatId)}>{chatId}</span>
              <div className="chat-buttons">
                <button onClick={(e) => renameChat(e, chatId)}>Rename</button>
                <button onClick={(e) => deleteChat(e, chatId)}>Delete</button>
              </div>
            </div>
          ))}
        </div>
        <button className="exit-button" onClick={() => window.close()}>Exit</button>
      </div>
      
      <div className="chat-container">
        <div className="messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              {msg.content}
            </div>
          ))}
        </div>
        
        <form onSubmit={sendMessage} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your question..."
          />
          <button type="submit">Send</button>
        </form>
      </div>
    </div>
  );
}

export default App;
