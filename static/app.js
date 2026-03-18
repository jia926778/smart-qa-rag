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
