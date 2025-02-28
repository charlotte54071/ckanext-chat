ckan.module("chat-module", function ($, _) {
  "use strict";
  return {
    options: {
      debug: false,
    },

    initialize: function () {},
  };
});


let defaulChatLabel = "Current Chat"; // Initialize current chat label
let currentChatLabel = defaulChatLabel;

function handleKeyDown(event) {
  if (event.key === 'Enter' && event.shiftKey) {
    const textarea = event.target;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;

    textarea.value = textarea.value.substring(0, start) + '\n' + textarea.value.substring(end);
    textarea.selectionStart = textarea.selectionEnd = start + 1;
  } else if (event.key === 'Enter') {
    event.preventDefault();
    sendMessage();
  }
}

function loadPreviousChats() {
  const chatListElement = document.getElementById('chatList');
  chatListElement.innerHTML = '';
  let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
  
  if (chats.length === 0) {
    chats.push({ title: currentChatLabel, messages: [] }); // Add default item
    localStorage.setItem('previousChats', JSON.stringify(chats));
  }

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

function loadChat(index) {
  let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
  if (chats[index]) {
    const chat = chats[index];
    const messagesDiv = document.getElementById('chatbox');
    messagesDiv.innerHTML = '';
    chat.messages.forEach(msg => {
      if (msg.kind === 'request') {
        // Append user message
        appendMessage('user', msg.parts);
      } else if (msg.kind === 'response') {
        // Append bot message
        appendMessage('bot', msg.parts);
      }
    });

    currentChatLabel = chat.title; // Update current chat label when loading a chat
  }
}

function getChatHistory(label = currentChatLabel) {
  let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
  return chats.find(chat => chat.title === label)?.messages || []; // Get messages based on label
}

function appendMessage(who, message) {
  var iconClass = who === 'user' ? 'fas fa-user' : 'fas fa-robot';

  // Check if message is an array of parts (from local storage)
  if (Array.isArray(message)) {
    message.forEach(part => {
      $('#chatbox').append(`
        <div class="message ${who === 'user' ? 'user-message' : 'bot-message'}">
          <span class="avatar"><i class="${iconClass}"></i></span>
          <div class="text">${part.content}</div>
        </div>
      `);
    });
  } else if (typeof message === 'object' && message !== null && message.parts) {
    // Check if message is an object with a parts array
    message.parts.forEach(part => {
      $('#chatbox').append(`
        <div class="message ${who === 'user' ? 'user-message' : 'bot-message'}">
          <span class="avatar"><i class="${iconClass}"></i></span>
          <div class="text">${part.content}</div>
        </div>
      `);
    });
  } else {
    // If it's just a plain string or non-parted message, append it directly
    $('#chatbox').append(`
      <div class="message ${who === 'user' ? 'user-message' : 'bot-message'}">
        <span class="avatar"><i class="${iconClass}"></i></span>
        <div class="text">${message}</div>
      </div>
    `);
  }

  $('#chatbox').scrollTop($('#chatbox')[0].scrollHeight); // Scroll to the bottom
}
function getLastEntryText(array) {
  const lastEntry = array[array.length - 1];
  if (lastEntry && lastEntry.parts && lastEntry.parts.length > 0) {
    const str=lastEntry.parts[lastEntry.parts.length - 1].content;
    return str.replace(/^"|"$/g, '').trim();
  }
  return null; // Return null if there is no valid entry
}

function sendMessage() {
  var text = $('#userInput').val();
  $('#userInput').val('');

  if (text.trim() !== '') {
    appendMessage('user', text);
    const chatHistory = getChatHistory(currentChatLabel);

    // If this is the first message, retitle the current chat
    if (chatHistory.length === 0) {
      currentChatLabel = "Current Chat"; 
    }

    $.post('chat/ask', { text: "Provide only a 3-word title for this question: " + text }, function(data) {
      const label = getLastEntryText(data.response);
      if (chatHistory.length === 0) {
        // Update the chat title in local storage
        updateChatTitle(currentChatLabel, label);
        currentChatLabel=label;
      }
      sendBotMessage(text, currentChatLabel); // Send message to the bot with the current chat label
    });
  }
}

function updateChatTitle(oldLabel, newLabel) {
  let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
  let chatIndex = chats.findIndex(chat => chat.title === oldLabel);

  if (chatIndex !== -1) {
    chats[chatIndex].title = newLabel; // Update the title
    localStorage.setItem('previousChats', JSON.stringify(chats));
    loadPreviousChats();
  }
}

function sendBotMessage(text, label) {
  const history = getChatHistory(label); // Load history based on the current label
  $.post('chat/ask', { text: text, history: JSON.stringify(history) }, function(data) {
    saveChat(data.response, label);
    appendMessage('bot', data.response[data.response.length - 1]);
  });
}

function saveChat(chat_messages, label) {
  let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
  
  let existingChatIndex = chats.findIndex(chat => chat.title === label);
  
  if (existingChatIndex === -1) {
    chats.push({ title: label, messages: [] });
  }

  let currentChat = chats[existingChatIndex === -1 ? chats.length - 1 : existingChatIndex];
  currentChat.messages = currentChat.messages.concat(chat_messages);
  
  localStorage.setItem('previousChats', JSON.stringify(chats));
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
  
  // Find the index of the chat with the current label
  const currentLabel = currentChatLabel; // Assuming currentChatLabel holds the label of the current chat
  const chatIndex = chats.findIndex(chat => chat.title === currentLabel);
  
  if (chatIndex !== -1) {
      // Remove the chat with the current label
      chats.splice(chatIndex, 1);
      localStorage.setItem('previousChats', JSON.stringify(chats));
      loadPreviousChats();
      document.getElementById('chatbox').innerHTML = '';
      currentChatLabel = defaulChatLabel; // Reset the current chat label if needed
  }
};
document.getElementById('newChatButton').onclick = function() {
  let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
  
  // Create a new empty chat with a default name
  chats.push({ title: defaulChatLabel, messages: [] }); // Create an empty chat

  localStorage.setItem('previousChats', JSON.stringify(chats)); // Save to local storage
  
  // Clear the chat area for a new chat
  document.getElementById('chatbox').innerHTML = '';
  $('#userInput').val('');
  currentChatLabel = defaulChatLabel; // Set current chat label to the new chat's name

  loadPreviousChats(); // Refresh the chat list in the UI
};
window.addEventListener('load', function() {
  let chats = JSON.parse(localStorage.getItem('previousChats')) || [];
  
  // Add the current chat only on the first load
  if (!chats.some(chat => chat.title === currentChatLabel)) {
    chats.push({ title: currentChatLabel, messages: [] });
    localStorage.setItem('previousChats', JSON.stringify(chats));
  }
  
  loadPreviousChats(); // Load previous chats after ensuring the current chat is added
});