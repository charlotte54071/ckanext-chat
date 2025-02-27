ckan.module("chat-module", function ($, _) {
  "use strict";
  return {
    options: {
      debug: false,
    },

    initialize: function () {},
  };
});

function handleKeyDown(event) {
  if (event.key === 'Enter' && event.shiftKey) {
      // Add a line break manually
      const textarea = event.target;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;

      // Insert a new line at the cursor position
      textarea.value = textarea.value.substring(0, start) + '\n' + textarea.value.substring(end);
      // Move the cursor to the new position
      textarea.selectionStart = textarea.selectionEnd = start + 1;
  } else if (event.key === 'Enter') {
      // Prevent the default action (form submission or other actions)
      event.preventDefault();
      sendMessage(); // Call your function to send the message
  }
}
// Load previous chats from localStorage
function loadPreviousChats() {
  const chatListElement = document.getElementById('chatList');
  chatListElement.innerHTML = '';
  let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
  chats.forEach((chat, index) => {
    let listItem = document.createElement('li');
    listItem.className = 'list-group-item list-group-item-action';
    listItem.textContent = chat.title || 'Chat ' + (index + 1);
    listItem.onclick = function () {
      loadChat(index);
    };
    chatListElement.appendChild(listItem);
  });
}

// Load a specific chat session from localStorage into the message area
function loadChat(index) {
  let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
  if (chats[index]) {
    const chat = chats[index];
    const messagesDiv = document.getElementById('chatbox');
    messagesDiv.innerHTML = '';
    chat.messages.forEach(msg => {
      appendMessage(msg.sender, msg.text);
    });
  }
}

function appendMessage(who, text) {
  var iconClass = who === 'user' ? 'fas fa-user' : 'fas fa-robot'; // Use Font Awesome classes for user and bot

  $('#chatbox').append(`
      <div class="message ${who === 'user' ? 'user-message' : 'bot-message'}">
          <span class="avatar"><i class="${iconClass}"></i></span>
          <div class="text">${text}</div>
      </div>
  `);
  $('#chatbox').scrollTop($('#chatbox')[0].scrollHeight); // Scroll to the bottom
}
var chatLabel = ""; // Flag to track if a label has been received from the chatbot

function sendMessage() {
  var text = $('#userInput').val();
  $('#userInput').val(''); // Clear the input after sending

  if (text.trim() !== '') {
    appendMessage('user', text);
    
    if (!chatLabel) {
      $.post('chat/ask', { text: "Provide only a 3-word title for this question: " + text }, function(data) {
        chatLabel = data.response.replace(/^"|"$/g, ''); // Store the label
        saveMessage('user', text, chatLabel); // Save the user message with the label
        sendBotMessage(text); // Send message to the bot
      });
    } else {
      saveMessage('user', text); // Save the user message with the existing label
      sendBotMessage(text); // Send message to the bot
    }
  }
}

function sendBotMessage(text) {
  $.post('chat/ask', { text: text }, function(data) {
    appendMessage('bot', data.response);
    saveMessage('bot', data.response); // Save the bot's response with the label
  });
}

// Update saveMessage function to ensure it handles labels correctly
function saveMessage(sender, text, label) {
    let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
    if (chats.length === 0 || (label && label.trim() !== '')) {
        // If a new label is provided, create a new chat with that label
        chats.push({ title: label || 'Chat ' + (chats.length + 1), messages: [] });
    }
    let currentChat = chats[chats.length - 1];
    currentChat.messages.push({ sender: sender, text: text });
    localStorage.setItem('previousChats', JSON.stringify(chats));
    loadPreviousChats();
}

function sendFile() {
  var file = $('#fileInput').prop('files')[0];
  var formData = new FormData();
  formData.append('file', file);

  $.ajax({
      url: '/upload',
      type: 'POST',
      data: formData,
      contentType: false,
      processData: false,
      success: function(data) {
          appendMessage('user', 'Uploaded a file');
          appendMessage('bot', data.response);
      }
  });
}
document.getElementById('deleteChatButton').onclick = function() {
  let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
  if (chats.length > 0) {
      chats.pop(); // Remove the last chat
      localStorage.setItem('previousChats', JSON.stringify(chats));
      loadPreviousChats(); // Refresh the chat list
      // Optionally clear the chat area
      document.getElementById('chatbox').innerHTML = '';
  }
};

document.getElementById('newChatButton').onclick = function() {
  // Clear the chat area for a new chat
  document.getElementById('chatbox').innerHTML = '';
  $('#userInput').val(''); // Clear input
  labelSet = false; // Reset label received flag
};
// Load previous chats when the page loads
window.addEventListener('DOMContentLoaded', loadPreviousChats);

