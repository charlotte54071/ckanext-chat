ckan.module("chat-module", function ($, _) {
  "use strict";
  _ = _ || window._; // use underscore if available

  // Private helper: Render Markdown and sanitize output
  function renderMarkdown(content) {
    var cleanHtml = "";
    if (Array.isArray(content)) {
      cleanHtml = content
        .map(function (item) {
          var rawHtml = marked.parse(item);
          return DOMPurify.sanitize(rawHtml, {
            ALLOWED_TAGS: [
              "p",
              "pre",
              "code",
              "span",
              "div",
              "br",
              "strong",
              "em",
              "ul",
              "ol",
              "li",
              "a",
            ],
            ALLOWED_ATTR: ["class", "href"],
          });
        })
        .join("");
    } else if (content) {
      var rawHtml = marked.parse(content);
      cleanHtml = DOMPurify.sanitize(rawHtml, {
        ALLOWED_TAGS: [
          "p",
          "pre",
          "code",
          "span",
          "div",
          "br",
          "strong",
          "em",
          "ul",
          "ol",
          "li",
          "a",
        ],
        ALLOWED_ATTR: ["class", "href"],
      });
    }
    return cleanHtml;
  }

  // Private helper: Convert timestamp strings to ISO format recursively
  function convertTimestampsToISO(data) {
    if (Array.isArray(data)) {
      return data.map(convertTimestampsToISO);
    } else if (typeof data === "object" && data !== null) {
      return Object.keys(data).reduce(function (acc, key) {
        acc[key] = convertTimestampsToISO(data[key]);
        if (key === "timestamp" && typeof acc[key] === "string") {
          acc[key] = new Date(acc[key]).toISOString();
        }
        return acc;
      }, {});
    }
    return data;
  }

  return {
    options: {
      debug: false,
    },
    currentChatLabel: "Current Chat",

    // Called automatically when the module is instantiated
    initialize: function () {
      this.bindUI();
      this.loadPreviousChats();
      if (this.options.debug) {
        console.log("Chat module initialized");
      }
      window.sendMessage = this.sendMessage.bind(this); // Bind sendMessage globally
    },

    // Bind all UI events within the module container and globally for sidebar elements
    bindUI: function () {
      var self = this;
      // Bind click events within the chat container
      this.el.find("#sendButton").on("click", function () {
        self.sendMessage();
      });
      this.el.find("#deleteChatButton").on("click", function () {
        self.deleteChat();
      });
      this.el.find("#newChatButton").on("click", function () {
        self.newChat();
      });
      this.el.find("#regenerateButton").on("click", function () {
        self.regenerateFailedMessage();
      });
      // Bind keydown event for the user input textarea
      this.el.find("#userInput").on("keydown", function (e) {
        self.handleKeyDown(e);
      });
      // Since the sidebar is rendered outside the module container, bind using a global selector
      $("#chatList").on("click", "li", function () {
        var index = $(this).index();
        self.loadChat(index);
      });
    },

    // Handler for keydown on the textarea
    handleKeyDown: function (event) {
      if (event.key === "Enter" && event.shiftKey) {
        var textarea = event.target;
        var start = textarea.selectionStart;
        var end = textarea.selectionEnd;
        textarea.value =
          textarea.value.substring(0, start) +
          "\n" +
          textarea.value.substring(end);
        textarea.selectionStart = textarea.selectionEnd = start + 1;
      } else if (event.key === "Enter") {
        event.preventDefault();
        this.sendMessage();
      }
    },

    // Load previous chats into the sidebar list
    loadPreviousChats: function () {
      var chatListElement = $("#chatList"); // Sidebar is outside the module container
      chatListElement.empty();
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      if (chats.length === 0) {
        chats.push({ title: this.currentChatLabel, messages: [] });
        localStorage.setItem("previousChats", JSON.stringify(chats));
      }
      var self = this;
      chats.forEach(function (chat, index) {
        var listItem = $("<li>")
          .addClass("list-group-item list-group-item-action")
          .text(chat.title || "Chat " + (index + 1))
          .on("click", function () {
            self.loadChat(index);
          });
        chatListElement.append(listItem);
      });
    },

    // Load a specific chat based on its index in localStorage
    loadChat: function (index) {
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      if (chats[index]) {
        var chat = chats[index];
        var messagesDiv = this.el.find("#chatbox");
        messagesDiv.empty();
        var self = this;
        chat.messages.forEach(function (msg) {
          if (msg.kind === "request") {
            self.appendMessage("user", msg.parts);
          } else if (msg.kind === "response") {
            self.appendMessage("bot", msg.parts);
          }
        });
        this.currentChatLabel = chat.title;
      }
    },

    // Retrieve chat history from localStorage and convert timestamps
    getChatHistory: function (label) {
      label = label || this.currentChatLabel || "Current Chat";
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      var storedData =
        (
          chats.find(function (chat) {
            return chat.title === label;
          }) || {}
        ).messages || [];
      return convertTimestampsToISO(storedData);
    },

    appendMessage: function (who, message) {
      var iconClass = who === "user" ? "fas fa-user" : "fas fa-robot";
      if (!Array.isArray(message)) {
        message = [{ content: message }];
      }
      var chatbox = this.el.find("#chatbox");
      var self = this;

      function formatContent(content) {
        if (typeof content === "object" && content !== null) {
          if (Array.isArray(content)) {
            return (
              "<ul>" +
              content
                .map(function (item) {
                  return "<li>" + formatContent(item) + "</li>";
                })
                .join("") +
              "</ul>"
            );
          } else {
            var html = "<ul>";
            for (var key in content) {
              if (content.hasOwnProperty(key)) {
                html +=
                  "<li>" + key + ": " + formatContent(content[key]) + "</li>";
              }
            }
            html += "</ul>";
            return html;
          }
        } else {
          return String(content);
        }
      }

      // Group tool-call/tool-return parts by tool_call_id.
      var toolGroups = {};
      var nonToolParts = [];

      message.forEach(function (part, idx) {
        if (part.part_kind === "system-prompt") return;
        if (who === "bot" && part.part_kind === "user-prompt") return;

        if (
          part.part_kind === "tool-call" ||
          part.part_kind === "tool-return"
        ) {
          var id = part.tool_call_id;
          if (!toolGroups[id]) {
            toolGroups[id] = { parts: [], order: idx };
          }
          toolGroups[id].parts.push(part);
        } else {
          nonToolParts.push({ part: part, order: idx });
        }
      });

      Object.keys(toolGroups)
        .sort(function (a, b) {
          return toolGroups[a].order - toolGroups[b].order;
        })
        .forEach(function (groupId) {
          var group = toolGroups[groupId].parts;
          var argumentsContent = "";
          var outputContent = "";
          var combinedTimestamp = group[0].timestamp;
          var toolName = group[0].tool_name;
          var succeeded = false;

          group.forEach(function (p) {
            if (new Date(p.timestamp) > new Date(combinedTimestamp)) {
              combinedTimestamp = p.timestamp;
            }
            if (p.part_kind === "tool-call") {
              argumentsContent += "<p>Arguments: " + p.args + "</p>";
            } else if (p.part_kind === "tool-return") {
              succeeded = true;
              outputContent += "<p>Output:</p>" + formatContent(p.content);
            }
          });

          var statusClass = succeeded ? "border-success" : "border-danger";
          var collapseId = "collapse" + groupId;
          var existingCollapse = chatbox.find("#" + collapseId);

          if (existingCollapse.length > 0) {
            var cardContainer = existingCollapse.closest(".col-auto.card");
            cardContainer
              .removeClass("border-danger border-success")
              .addClass(statusClass);
            cardContainer
              .find(".card-title")
              .html("Tool Call: " + toolName + " " + combinedTimestamp);
            var cardBody = existingCollapse.find(".card-body");
            cardBody.append(argumentsContent + outputContent);
          } else {
            var combinedCardHtml = $(`
              <div class="message bot-message">
                <span class="col-2 chatavatar"><i class="fas fa-robot"></i></span>
                <div class="col-auto card text ${statusClass}" style="cursor:pointer;">
                  <div class="card-body p-0">
                    <h5 class="card-title">Tool Call: ${toolName} ${combinedTimestamp}</h5>
                    <div class="collapse mt-2" id="${collapseId}">
                      <div class="card card-body">
                        ${argumentsContent}
                        ${outputContent}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            `);
            combinedCardHtml.on("click", function () {
              self.toggleDetails(collapseId);
            });
            chatbox.append(combinedCardHtml);
          }
        });

      nonToolParts.sort(function (a, b) {
        return a.order - b.order;
      });
      nonToolParts.forEach(function (item, index) {
        var part = item.part;
        var messageHtml = $(`
          <div class="message ${who === "user" ? "user-message" : "bot-message"}">
            <span class="col-2 chatavatar"><i class="${iconClass}"></i></span>
            <div class="col-auto text">
              ${renderMarkdown(part.content)}
            </div>
          </div>
        `);

        if (who === "bot") {
          messageHtml.find("a").each(function () {
            var link = $(this);
            var buttonId = `download-btn-${index}`; // Generate a unique ID for each button
            var downloadButton = $(
              '<button id="${buttonId}" class="btn download-link-content-btn" data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="import file contents to prompt"><i class="fas fa-file-import"></i></button>',
            );

            downloadButton.on("click", function () {
              var url = link.attr("href");
              $.get(url, function (data) {
                self.el.find("#userInput").val(data); // Paste the content into the textarea
                $("html, body").animate(
                  { scrollTop: $(document).height() },
                  "slow",
                ); // Scroll to the bottom of the page
              }).fail(function () {
                alert("Failed to download content from the URL.");
              });
            });

            link.after(downloadButton);
          });
        }

        chatbox.append(messageHtml);
      });

      chatbox.find("pre code").each(function () {
        if (!$(this).attr("data-highlighted")) {
          hljs.highlightElement(this);
          $(this).attr("data-highlighted", "true");
        }
      });
      self.addCopyButtonsToCodeBlocks();
      chatbox.scrollTop(chatbox[0].scrollHeight);

      // Reinitialize tooltips for dynamically added buttons
      $('[data-bs-toggle="tooltip"]').tooltip();
    },

    // Toggle collapsible details for tool messages.
    toggleDetails: function (collapseId) {
      var collapseElement = document.getElementById(collapseId);
      if (collapseElement) {
        new bootstrap.Collapse(collapseElement, { toggle: true });
      }
    },

    // Add copy buttons to code blocks
    addCopyButtonsToCodeBlocks: function () {
      this.el.find("pre code").each(function () {
        var codeBlock = $(this);
        if (codeBlock.parent().find(".copy-button").length > 0) return;
        var copyButton = $(
          '<button class="copy-button"><i class="fas fa-copy"></i></button>',
        );
        copyButton.on("click", function () {
          navigator.clipboard
            .writeText(codeBlock.text())
            .then(function () {
              copyButton.html('<i class="fas fa-check"></i>');
              setTimeout(function () {
                copyButton.html('<i class="fas fa-copy"></i>');
              }, 2000);
            })
            .catch(function (err) {
              console.error("Failed to copy: ", err);
            });
        });
        var preElement = codeBlock.parent();
        preElement.css("position", "relative");
        copyButton.css({ position: "absolute", top: "5px", right: "5px" });
        preElement.append(copyButton);
      });
    },

    // Retrieve text from the last entry (used for renaming the chat)
    getLastEntryText: function (array) {
      var lastEntry = array[array.length - 1];
      if (lastEntry && lastEntry.parts && lastEntry.parts.length > 0) {
        var str = lastEntry.parts[lastEntry.parts.length - 1].content;
        return str.replace(/^"|"$/g, "").trim();
      }
      return null;
    },

    // Send a user message and then trigger a bot reply
    sendMessage: function () {
      var self = this;
      var text = this.el.find("#userInput").val();
      this.el.find("#userInput").val("");
      if (text.trim() !== "") {
        self.appendMessage("user", text);
        var chatHistory = self.getChatHistory();
        if (!chatHistory.length) {
          self.currentChatLabel = "Current Chat";
        }
        var sendButton = this.el.find("#sendButton");
        var spinner = sendButton.find(".spinner-border");
        var buttonText = sendButton.find(".button-text");
        var icon = sendButton.find(".fa-paper-plane");
        sendButton.prop("disabled", true);
        spinner.removeClass("d-none");
        buttonText.addClass("d-none");
        icon.addClass("d-none");

        $.post("chat/ask", {
          text: "Provide only a 3-word title for this question: " + text,
        })
          .done(function (data) {
            var label = self.getLastEntryText(data.response);
            if (!chatHistory.length) {
              self.updateChatTitle(self.currentChatLabel, label);
              self.currentChatLabel = label;
            }
            self.sendBotMessage(text, self.currentChatLabel, function () {
              spinner.addClass("d-none");
              buttonText.removeClass("d-none");
              icon.removeClass("d-none");
              sendButton.prop("disabled", false);
            });
          })
          .fail(function () {
            alert("An error occurred while processing your request.");
            spinner.addClass("d-none");
            buttonText.removeClass("d-none");
            icon.removeClass("d-none");
            sendButton.prop("disabled", false);
          });
      }
    },

    // Update the chat title in localStorage
    updateChatTitle: function (oldLabel, newLabel) {
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      var chatIndex = chats.findIndex(function (chat) {
        return chat.title === oldLabel;
      });
      if (chatIndex !== -1) {
        chats[chatIndex].title = newLabel;
        localStorage.setItem("previousChats", JSON.stringify(chats));
        this.loadPreviousChats();
      }
    },

    // Send a request to the bot and append its reply
    sendBotMessage: function (text, label, callback) {
      var history = this.getChatHistory(label);
      var self = this;
      $.post(
        "chat/ask",
        { text: text, history: JSON.stringify(history) },
        function (data) {
          self.saveChat(data.response, label);
          data.response.forEach(function (msg, index) {
            // Skip the first element by checking the index because its the lat user prompt
            // if (index === 0) return;
            // console.log(msg)
            if (msg.kind === "request") {
              self.appendMessage("bot", msg.parts);
            } else if (msg.kind === "response") {
              self.appendMessage("bot", msg.parts);
            }
          });
          // self.appendMessage("bot", data.response[data.response.length - 1].parts);
          if (callback) callback();
        },
      );
    },

    // Save new messages to the chat history in localStorage
    saveChat: function (newMessages, label) {
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      var existingChatIndex = chats.findIndex(function (chat) {
        return chat.title === label;
      });
      if (existingChatIndex === -1) {
        chats.push({ title: label, messages: [] });
        existingChatIndex = chats.length - 1;
      }
      chats[existingChatIndex].messages =
        chats[existingChatIndex].messages.concat(newMessages);
      localStorage.setItem("previousChats", JSON.stringify(chats));
    },

    // Regenerate the failed message (if any)
    regenerateFailedMessage: function () {
      var currentLabel = this.currentChatLabel || "Current Chat";
      // Retrieve chat history using your helper
      var chatHistory = this.getChatHistory(currentLabel);
      if (!chatHistory.length) {
        alert("No chat history available.");
        return;
      }

      // Locate the last message containing a part with "user-prompt"
      var lastUserIndex = -1;
      for (var i = chatHistory.length - 1; i >= 0; i--) {
        if (
          chatHistory[i].parts &&
          chatHistory[i].parts.some(function (part) {
            return part.part_kind === "user-prompt";
          })
        ) {
          lastUserIndex = i;
          break;
        }
      }
      if (lastUserIndex === -1) {
        alert("No user prompt found in chat history.");
        return;
      }

      // Remove the last user prompt message from the stored chat history
      var newChatHistory = chatHistory.slice(0, lastUserIndex);

      // Update localStorage with the trimmed chat history
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      var chatIndex = chats.findIndex(function (chat) {
        return chat.title === currentLabel;
      });
      if (chatIndex === -1) {
        alert("Current chat not found in storage.");
        return;
      }
      chats[chatIndex].messages = newChatHistory;
      localStorage.setItem("previousChats", JSON.stringify(chats));

      // Refresh the chat UI
      this.el.find("#chatbox").empty();
      this.loadChat(chatIndex);

      // Retrieve the user prompt text from the removed message
      var lastUserMessage = chatHistory[lastUserIndex];
      var userPromptPart = lastUserMessage.parts.find(function (part) {
        return part.part_kind === "user-prompt";
      });
      var userText = userPromptPart ? String(userPromptPart.content) : "";
      if (!userText.trim()) {
        alert("User prompt text is empty.");
        return;
      }

      // Set the user input field and call sendMessage (which triggers the spinner)
      this.el.find("#userInput").val(userText);
      this.sendMessage();
    },

    // Send a file via AJAX (if needed)
    sendFile: function () {
      var file = this.el.find("#fileInput").prop("files")[0];
      var formData = new FormData();
      formData.append("file", file);
      $.ajax({
        url: "/upload",
        type: "POST",
        data: formData,
        contentType: false,
        processData: false,
        success: (data) => {
          this.appendMessage("user", "Uploaded a file");
          this.appendMessage("bot", data.response);
        },
      });
    },

    // Delete the current chat and update localStorage
    deleteChat: function () {
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      var currentLabel = this.currentChatLabel || "Current Chat";
      var chatIndex = chats.findIndex(function (chat) {
        return chat.title === currentLabel;
      });
      if (chatIndex !== -1) {
        chats.splice(chatIndex, 1);
        localStorage.setItem("previousChats", JSON.stringify(chats));
        this.loadPreviousChats();
        this.el.find("#chatbox").empty();
        this.currentChatLabel = "Current Chat";
      }
    },

    // Start a new chat session
    newChat: function () {
      var chats = JSON.parse(localStorage.getItem("previousChats")) || [];
      chats.push({ title: "Current Chat", messages: [] });
      localStorage.setItem("previousChats", JSON.stringify(chats));
      this.el.find("#chatbox").empty();
      this.el.find("#userInput").val("");
      this.currentChatLabel = "Current Chat";
      this.loadPreviousChats();
    },
  };
});
