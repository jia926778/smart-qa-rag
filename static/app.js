/* ================================================================
   Smart QA RAG - Frontend Logic
   ================================================================ */

const API = {
  ask: "/api/v1/qa/ask",
  upload: "/api/v1/documents/upload",
  listDocs: (col) => `/api/v1/documents/${encodeURIComponent(col)}`,
  deleteDoc: (col, src) =>
    `/api/v1/documents/${encodeURIComponent(col)}/${encodeURIComponent(src)}`,
  createCollection: "/api/v1/collections/",
  listCollections: "/api/v1/collections/",
  deleteCollection: (name) =>
    `/api/v1/collections/${encodeURIComponent(name)}`,
};

// ---- DOM refs ----
const messagesEl = document.getElementById("messages");
const questionInput = document.getElementById("questionInput");
const btnSend = document.getElementById("btnSend");
const collectionSelect = document.getElementById("collectionSelect");
const currentCollectionBadge = document.getElementById("currentCollection");
const newCollectionName = document.getElementById("newCollectionName");
const btnCreateCollection = document.getElementById("btnCreateCollection");
const btnDeleteCollection = document.getElementById("btnDeleteCollection");
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const docList = document.getElementById("docList");
const btnToggleSidebar = document.getElementById("btnToggleSidebar");
const sidebar = document.getElementById("sidebar");

// ---- State ----
let chatHistory = [];

// ---- Helpers ----
function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setUploadStatus(msg, isError) {
  uploadStatus.textContent = msg;
  uploadStatus.className = "upload-status " + (isError ? "error" : "success");
}

function getCollection() {
  return collectionSelect.value || "default";
}

// ---- Markdown rendering ----
function renderMarkdown(text) {
  if (typeof marked !== "undefined") {
    return marked.parse(text);
  }
  return text.replace(/\n/g, "<br>");
}

// ---- Message rendering ----
function addMessage(role, content, sources, elapsedMs) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdown(content);
  wrapper.appendChild(bubble);

  if (sources && sources.length > 0) {
    const toggle = document.createElement("div");
    toggle.className = "sources-toggle";
    toggle.textContent = `📎 ${sources.length} 个参考来源 (点击展开)`;

    const panel = document.createElement("div");
    panel.className = "sources-panel";
    sources.forEach((s, i) => {
      const item = document.createElement("div");
      item.className = "source-item";
      let label = `<strong>[${i + 1}] ${s.source}`;
      if (s.page != null) label += ` - 第${s.page}页`;
      label += `</strong><br>${s.content}`;
      item.innerHTML = label;
      panel.appendChild(item);
    });

    toggle.addEventListener("click", () => {
      panel.classList.toggle("open");
      toggle.textContent = panel.classList.contains("open")
        ? `📎 ${sources.length} 个参考来源 (点击收起)`
        : `📎 ${sources.length} 个参考来源 (点击展开)`;
    });

    bubble.appendChild(toggle);
    bubble.appendChild(panel);
  }

  if (elapsedMs != null) {
    const el = document.createElement("div");
    el.className = "elapsed";
    el.textContent = `耗时 ${elapsedMs.toFixed(0)} ms`;
    bubble.appendChild(el);
  }

  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function addLoading() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  wrapper.id = "loading-msg";
  const bubble = document.createElement("div");
  bubble.className = "bubble loading-dots";
  bubble.innerHTML = "<span></span><span></span><span></span>";
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

function removeLoading() {
  const el = document.getElementById("loading-msg");
  if (el) el.remove();
}

// ---- API calls ----
async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  questionInput.style.height = "auto";

  chatHistory.push({ role: "user", content: question });

  addLoading();
  btnSend.disabled = true;

  try {
    const resp = await fetch(API.ask, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        collection_name: getCollection(),
        chat_history: chatHistory.slice(-6),
      }),
    });
    removeLoading();

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      addMessage("assistant", `抱歉，请求出错：${err.detail || resp.statusText}`);
      return;
    }

    const data = await resp.json();
    addMessage("assistant", data.answer, data.sources, data.elapsed_ms);
    chatHistory.push({ role: "assistant", content: data.answer });
  } catch (e) {
    removeLoading();
    addMessage("assistant", `网络错误：${e.message}`);
  } finally {
    btnSend.disabled = false;
    questionInput.focus();
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("collection_name", getCollection());

  setUploadStatus("正在上传并处理...", false);

  try {
    const resp = await fetch(API.upload, { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      setUploadStatus(err.detail || "上传失败", true);
      return;
    }
    const data = await resp.json();
    setUploadStatus(`${data.filename} 已上传 (${data.chunks_count} 个分块)`, false);
    loadDocuments();
  } catch (e) {
    setUploadStatus(`上传失败：${e.message}`, true);
  }
}

async function loadCollections() {
  try {
    const resp = await fetch(API.listCollections);
    if (!resp.ok) return;
    const list = await resp.json();
    const current = collectionSelect.value;
    collectionSelect.innerHTML = "";
    if (list.length === 0) {
      collectionSelect.innerHTML = '<option value="default">default</option>';
    } else {
      list.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c.name;
        opt.textContent = `${c.name} (${c.documents_count})`;
        collectionSelect.appendChild(opt);
      });
    }
    if ([...collectionSelect.options].some((o) => o.value === current)) {
      collectionSelect.value = current;
    }
    currentCollectionBadge.textContent = getCollection();
  } catch (e) {
    /* ignore */
  }
}

async function loadDocuments() {
  const col = getCollection();
  try {
    const resp = await fetch(API.listDocs(col));
    if (!resp.ok) {
      docList.innerHTML = '<p class="empty-hint">暂无文档</p>';
      return;
    }
    const docs = await resp.json();
    if (docs.length === 0) {
      docList.innerHTML = '<p class="empty-hint">暂无文档</p>';
      return;
    }
    docList.innerHTML = "";
    docs.forEach((d) => {
      const item = document.createElement("div");
      item.className = "doc-item";
      item.innerHTML = `<span class="doc-name" title="${d.source}">${d.source} (${d.chunks})</span>`;
      const btn = document.createElement("button");
      btn.textContent = "删除";
      btn.addEventListener("click", async () => {
        if (!confirm(`确定删除 ${d.source}？`)) return;
        await fetch(API.deleteDoc(col, d.source), { method: "DELETE" });
        loadDocuments();
        loadCollections();
      });
      item.appendChild(btn);
      docList.appendChild(item);
    });
  } catch (e) {
    docList.innerHTML = '<p class="empty-hint">加载失败</p>';
  }
}

// ---- Events ----
btnSend.addEventListener("click", sendQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

// Auto-resize textarea
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + "px";
});

// Collection events
collectionSelect.addEventListener("change", () => {
  currentCollectionBadge.textContent = getCollection();
  loadDocuments();
});

btnCreateCollection.addEventListener("click", async () => {
  const name = newCollectionName.value.trim();
  if (!name) return;
  try {
    const resp = await fetch(API.createCollection, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description: "" }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert(err.detail || "创建失败");
      return;
    }
    newCollectionName.value = "";
    await loadCollections();
    collectionSelect.value = name;
    currentCollectionBadge.textContent = name;
    loadDocuments();
  } catch (e) {
    alert("创建失败：" + e.message);
  }
});

btnDeleteCollection.addEventListener("click", async () => {
  const name = getCollection();
  if (!confirm(`确定删除知识库 "${name}"？所有文档将被移除。`)) return;
  try {
    await fetch(API.deleteCollection(name), { method: "DELETE" });
    await loadCollections();
    loadDocuments();
  } catch (e) {
    alert("删除失败：" + e.message);
  }
});

// Upload events
uploadArea.addEventListener("click", () => fileInput.click());
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});
uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    uploadFile(fileInput.files[0]);
    fileInput.value = "";
  }
});

// Sidebar toggle
btnToggleSidebar.addEventListener("click", () => {
  sidebar.classList.toggle("hidden");
});

// ---- Init ----
loadCollections();
loadDocuments();

/* ================================================================
   新增功能模块
   ================================================================ */

// ---- NotificationManager (通知管理器) ----
const NotificationManager = {
  container: document.getElementById("notificationContainer"),
  
  show(message, type = "info", duration = 3000) {
    const notification = document.createElement("div");
    notification.className = `notification ${type}`;
    notification.textContent = message;
    this.container.appendChild(notification);
    
    if (duration > 0) {
      setTimeout(() => {
        notification.style.animation = "slideIn 0.3s ease reverse";
        setTimeout(() => notification.remove(), 300);
      }, duration);
    }
    
    return notification;
  },
  
  showSuccess(message, duration = 2000) {
    return this.show(message, "success", duration);
  },
  
  showError(message, duration = 4000) {
    return this.show(message, "error", duration);
  },
  
  showInfo(message, duration = 3000) {
    return this.show(message, "info", duration);
  }
};

// ---- StorageManager (存储管理器) ----
const StorageManager = {
  maxChatHistory: 50,
  maxSize: 5 * 1024 * 1024,
  
  save(key, data) {
    try {
      const serialized = JSON.stringify(data);
      if (serialized.length > this.maxSize) {
        this.pruneOldData(key);
      }
      localStorage.setItem(key, serialized);
    } catch (e) {
      console.error("Storage save failed:", e);
    }
  },
  
  load(key) {
    try {
      const data = localStorage.getItem(key);
      return data ? JSON.parse(data) : null;
    } catch (e) {
      return null;
    }
  },
  
  clear(key) {
    localStorage.removeItem(key);
  },
  
  pruneOldData(key) {
    const data = this.load(key);
    if (data && data.messages && data.messages.length > this.maxChatHistory) {
      data.messages = data.messages.slice(-this.maxChatHistory);
      this.save(key, data);
    }
  }
};

// ---- ThemeManager (主题管理器) ----
const ThemeManager = {
  currentTheme: "light",
  storageKey: "theme_preference",
  btnToggleTheme: document.getElementById("btnToggleTheme"),
  
  init() {
    this.loadPreference();
    this.btnToggleTheme.addEventListener("click", () => this.toggleTheme());
  },
  
  toggleTheme() {
    this.currentTheme = this.currentTheme === "light" ? "dark" : "light";
    this.applyTheme();
    this.savePreference();
    this.updateButtonIcon();
  },
  
  applyTheme() {
    document.documentElement.setAttribute("data-theme", this.currentTheme);
  },
  
  savePreference() {
    localStorage.setItem(this.storageKey, this.currentTheme);
  },
  
  loadPreference() {
    const saved = localStorage.getItem(this.storageKey);
    if (saved) {
      this.currentTheme = saved;
      this.applyTheme();
    }
    this.updateButtonIcon();
  },
  
  updateButtonIcon() {
    this.btnToggleTheme.textContent = this.currentTheme === "light" ? "🌙" : "☀️";
  }
};

// ---- ChatManager (对话管理器) ----
const ChatManager = {
  btnClearChat: document.getElementById("btnClearChat"),
  btnExportChat: document.getElementById("btnExportChat"),
  storageKey: "chat_history",
  
  init() {
    this.btnClearChat.addEventListener("click", () => this.clearChat());
    this.btnExportChat.addEventListener("click", () => this.exportChat());
    this.loadFromStorage();
  },
  
  clearChat() {
    if (!confirm("确定要清空当前对话吗？")) return;
    
    messagesEl.innerHTML = "";
    chatHistory = [];
    
    // 显示欢迎消息
    const welcomeMsg = document.createElement("div");
    welcomeMsg.className = "message assistant";
    welcomeMsg.innerHTML = `<div class="bubble">您好！我是智能问答助手，请先在左侧上传文档到知识库，然后向我提问吧。</div>`;
    messagesEl.appendChild(welcomeMsg);
    
    StorageManager.clear(this.storageKey);
    NotificationManager.showSuccess("对话已清空");
  },
  
  exportChat() {
    if (chatHistory.length === 0) {
      NotificationManager.showError("暂无对话记录可导出");
      return;
    }
    
    const collection = getCollection();
    const now = new Date();
    const timestamp = now.toLocaleString("zh-CN");
    
    let markdown = `# Smart QA 对话记录\n\n`;
    markdown += `**导出时间**: ${timestamp}\n`;
    markdown += `**知识库**: ${collection}\n\n---\n\n`;
    
    chatHistory.forEach((msg, idx) => {
      const role = msg.role === "user" ? "用户" : "AI 助手";
      markdown += `## ${role}\n${msg.content}\n\n`;
      if (msg.sources && msg.sources.length > 0) {
        markdown += `**来源**:\n`;
        msg.sources.forEach((s, i) => {
          markdown += `- [${i + 1}] ${s.source}`;
          if (s.page != null) markdown += ` - 第${s.page}页`;
          markdown += `\n`;
        });
        markdown += `\n`;
      }
      markdown += `---\n\n`;
    });
    
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat_${collection}_${now.getTime()}.md`;
    a.click();
    URL.revokeObjectURL(url);
    
    NotificationManager.showSuccess("对话已导出");
  },
  
  async copyMessage(content) {
    try {
      await navigator.clipboard.writeText(content);
      NotificationManager.showSuccess("已复制到剪贴板");
    } catch (e) {
      // 降级方案
      const textarea = document.createElement("textarea");
      textarea.value = content;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      NotificationManager.showSuccess("已复制到剪贴板");
    }
  },
  
  saveToStorage() {
    const data = {
      collection_name: getCollection(),
      messages: chatHistory,
      last_updated: Date.now()
    };
    StorageManager.save(this.storageKey, data);
  },
  
  loadFromStorage() {
    const data = StorageManager.load(this.storageKey);
    if (data && data.messages && data.messages.length > 0) {
      // 恢复对话历史
      data.messages.forEach(msg => {
        if (msg.role === "user") {
          addMessage("user", msg.content);
        } else {
          addMessage("assistant", msg.content, msg.sources);
        }
      });
      chatHistory = data.messages;
    }
  }
};

// ---- UploadManager (上传管理器) ----
const UploadManager = {
  uploadProgress: document.getElementById("uploadProgress"),
  maxFiles: 10,
  
  getFileIcon(filename) {
    const ext = filename.split(".").pop().toLowerCase();
    const iconMap = {
      pdf: "📄", docx: "📝", doc: "📝", xlsx: "📊", xls: "📊",
      pptx: "📽️", ppt: "📽️", md: "📑", txt: "📃",
      html: "🌐", htm: "🌐", json: "📋", xml: "📋",
      yaml: "⚙️", yml: "⚙️", csv: "📊",
      png: "🖼️", jpg: "🖼️", jpeg: "🖼️", gif: "🖼️",
      mp3: "🎵", wav: "🎵", mp4: "🎬", webm: "🎬"
    };
    return iconMap[ext] || "📁";
  },
  
  async uploadMultiple(files) {
    const fileArray = Array.from(files).slice(0, this.maxFiles);
    if (fileArray.length < files.length) {
      NotificationManager.showInfo(`最多同时上传 ${this.maxFiles} 个文件`);
    }
    
    this.uploadProgress.innerHTML = "";
    
    for (let i = 0; i < fileArray.length; i++) {
      await this.uploadFileWithProgress(fileArray[i], i, fileArray.length);
    }
    
    loadDocuments();
    loadCollections();
  },
  
  async uploadFileWithProgress(file, index, total) {
    const progressId = `progress-${Date.now()}-${index}`;
    
    // 创建进度条
    const progressItem = document.createElement("div");
    progressItem.className = "progress-item";
    progressItem.id = progressId;
    progressItem.innerHTML = `
      <div class="filename">
        <span>${this.getFileIcon(file.name)} ${file.name}</span>
        <span class="percent">0%</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: 0%"></div>
      </div>
    `;
    this.uploadProgress.appendChild(progressItem);
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("collection_name", getCollection());
    
    // 模拟进度更新
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress > 90) progress = 90;
      this.updateProgress(progressId, progress);
    }, 200);
    
    try {
      const resp = await fetch(API.upload, { method: "POST", body: formData });
      clearInterval(progressInterval);
      
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        this.updateProgress(progressId, 100, "失败");
        NotificationManager.showError(`${file.name}: ${err.detail || "上传失败"}`);
        setTimeout(() => progressItem.remove(), 2000);
        return;
      }
      
      const data = await resp.json();
      this.updateProgress(progressId, 100, "完成");
      NotificationManager.showSuccess(`${file.name} 已上传 (${data.chunks_count} 个分块)`);
      setTimeout(() => progressItem.remove(), 1500);
      
    } catch (e) {
      clearInterval(progressInterval);
      this.updateProgress(progressId, 100, "失败");
      NotificationManager.showError(`${file.name}: ${e.message}`);
      setTimeout(() => progressItem.remove(), 2000);
    }
  },
  
  updateProgress(id, percent, status) {
    const item = document.getElementById(id);
    if (!item) return;
    
    const fill = item.querySelector(".progress-fill");
    const percentText = item.querySelector(".percent");
    
    fill.style.width = `${percent}%`;
    percentText.textContent = status || `${Math.round(percent)}%`;
  }
};

// ---- KeyboardShortcuts (键盘快捷键) ----
const KeyboardShortcuts = {
  init() {
    document.addEventListener("keydown", (e) => this.handleShortcut(e));
  },
  
  handleShortcut(e) {
    // Ctrl/Cmd + L: 清空对话
    if ((e.ctrlKey || e.metaKey) && e.key === "l") {
      e.preventDefault();
      ChatManager.clearChat();
    }
    
    // Ctrl/Cmd + N: 聚焦输入框
    if ((e.ctrlKey || e.metaKey) && e.key === "n") {
      e.preventDefault();
      questionInput.focus();
    }
    
    // Ctrl/Cmd + E: 导出对话
    if ((e.ctrlKey || e.metaKey) && e.key === "e") {
      e.preventDefault();
      ChatManager.exportChat();
    }
  }
};

// ---- 修改现有函数以支持新功能 ----

// 修改 addMessage 函数添加复制按钮
const originalAddMessage = addMessage;
addMessage = function(role, content, sources, elapsedMs) {
  const wrapper = originalAddMessage(role, content, sources, elapsedMs);
  
  // 为 assistant 消息添加复制按钮
  if (role === "assistant" && wrapper) {
    const bubble = wrapper.querySelector(".bubble");
    if (bubble) {
      bubble.style.position = "relative";
      
      const copyBtn = document.createElement("button");
      copyBtn.className = "copy-btn";
      copyBtn.textContent = "📋";
      copyBtn.title = "复制";
      copyBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        ChatManager.copyMessage(content);
      });
      bubble.appendChild(copyBtn);
    }
  }
  
  return wrapper;
};

// 修改 sendQuestion 函数保存对话历史
const originalSendQuestion = sendQuestion;
sendQuestion = async function() {
  const question = questionInput.value.trim();
  if (!question) return;
  
  addMessage("user", question);
  questionInput.value = "";
  questionInput.style.height = "auto";
  
  chatHistory.push({ role: "user", content: question });
  
  addLoading();
  btnSend.disabled = true;
  
  try {
    const resp = await fetch(API.ask, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        collection_name: getCollection(),
        chat_history: chatHistory.slice(-6),
      }),
    });
    removeLoading();
    
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      const errorMsg = `抱歉，请求出错：${err.detail || resp.statusText}`;
      addMessage("assistant", errorMsg);
      NotificationManager.showError(err.detail || resp.statusText);
      return;
    }
    
    const data = await resp.json();
    addMessage("assistant", data.answer, data.sources, data.elapsed_ms);
    chatHistory.push({ 
      role: "assistant", 
      content: data.answer,
      sources: data.sources
    });
    
    // 保存对话历史
    ChatManager.saveToStorage();
    
  } catch (e) {
    removeLoading();
    const errorMsg = `网络错误：${e.message}`;
    addMessage("assistant", errorMsg);
    NotificationManager.showError(e.message);
  } finally {
    btnSend.disabled = false;
    questionInput.focus();
  }
};

// 修改文件上传支持批量
const originalUploadFile = uploadFile;
uploadFile = async function(file) {
  await UploadManager.uploadMultiple([file]);
};

// 修改拖拽事件支持多文件
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    UploadManager.uploadMultiple(e.dataTransfer.files);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    UploadManager.uploadMultiple(fileInput.files);
    fileInput.value = "";
  }
});

// 修改创建知识库添加成功提示
const originalBtnCreateCollectionHandler = btnCreateCollection.onclick;
btnCreateCollection.addEventListener("click", async () => {
  const name = newCollectionName.value.trim();
  if (!name) return;
  try {
    const resp = await fetch(API.createCollection, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description: "" }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      NotificationManager.showError(err.detail || "创建失败");
      return;
    }
    newCollectionName.value = "";
    await loadCollections();
    collectionSelect.value = name;
    currentCollectionBadge.textContent = name;
    loadDocuments();
    NotificationManager.showSuccess(`知识库 "${name}" 创建成功`);
  } catch (e) {
    NotificationManager.showError("创建失败：" + e.message);
  }
});

// 修改删除知识库添加成功提示
btnDeleteCollection.addEventListener("click", async () => {
  const name = getCollection();
  if (!confirm(`确定删除知识库 "${name}"？所有文档将被移除。`)) return;
  try {
    await fetch(API.deleteCollection(name), { method: "DELETE" });
    await loadCollections();
    loadDocuments();
    NotificationManager.showSuccess(`知识库 "${name}" 已删除`);
  } catch (e) {
    NotificationManager.showError("删除失败：" + e.message);
  }
});

// 修改文档列表显示文件图标
const originalLoadDocuments = loadDocuments;
loadDocuments = async function() {
  const col = getCollection();
  try {
    const resp = await fetch(API.listDocs(col));
    if (!resp.ok) {
      docList.innerHTML = '<p class="empty-hint">暂无文档</p>';
      return;
    }
    const docs = await resp.json();
    if (docs.length === 0) {
      docList.innerHTML = '<p class="empty-hint">暂无文档</p>';
      return;
    }
    docList.innerHTML = "";
    docs.forEach((d) => {
      const item = document.createElement("div");
      item.className = "doc-item";
      const icon = UploadManager.getFileIcon(d.source);
      item.innerHTML = `<span class="doc-name" title="${d.source}"><span class="file-icon">${icon}</span>${d.source} (${d.chunks})</span>`;
      const btn = document.createElement("button");
      btn.textContent = "删除";
      btn.addEventListener("click", async () => {
        if (!confirm(`确定删除 ${d.source}？`)) return;
        await fetch(API.deleteDoc(col, d.source), { method: "DELETE" });
        loadDocuments();
        loadCollections();
        NotificationManager.showSuccess(`${d.source} 已删除`);
      });
      item.appendChild(btn);
      docList.appendChild(item);
    });
  } catch (e) {
    docList.innerHTML = '<p class="empty-hint">加载失败</p>';
  }
};

// ---- 初始化所有新功能 ----
ThemeManager.init();
ChatManager.init();
KeyboardShortcuts.init();
