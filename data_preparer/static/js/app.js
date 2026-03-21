(function () {
  const $ = (id) => document.getElementById(id);

  // ==================== State ====================
  const state = {
    // Upload tab
    uploadToken: null,
    segments: [],
    selected: new Set(),
    maxSelect: 3,
    categories: [],
    // Annotate tab
    sessions: [],
    sessionIdx: -1,
    pointType: "human",
    points: { human: [], object: [] },
    annotFrameDataUrl: null,
    annotFrameSize: { w: 0, h: 0 },
    annotFrameTimestamp: null,
    cursorTimestamp: 0,
    selectedTimestamp: null,
  };

  // ==================== Helpers ====================
  function toast(msg) {
    const el = $("toast");
    if (!el) return;
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.classList.remove("show"), 4000);
  }

  function setStatus(el, msg, type) {
    if (!el) return;
    el.textContent = msg || "";
    el.className = "status" + (type ? " " + type : "");
  }

  async function apiGet(url) {
    const r = await fetch(url, { cache: "no-store" });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || r.statusText);
    return d;
  }

  async function apiPost(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || r.statusText);
    return d;
  }

  async function apiPostForm(url, formData) {
    const r = await fetch(url, { method: "POST", body: formData });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || r.statusText);
    return d;
  }

  function escapeHtml(s) {
    return (s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
    );
  }

  // ==================== Tab Switching ====================
  function switchTab(tab) {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".page").forEach((p) => (p.style.display = "none"));
    $("tab" + tab.charAt(0).toUpperCase() + tab.slice(1)).classList.add("active");
    $("page" + tab.charAt(0).toUpperCase() + tab.slice(1)).style.display = "";

    if (tab === "annotate") refreshSessions();
  }

  // ==================== Upload Tab ====================
  function initUploadTab() {
    const objectCategoryInput = $("objectCategory");
    const categorySuggest = $("categorySuggest");
    const videoFileInput = $("videoFile");
    const dropZone = $("dropZone");
    const dropFileName = $("dropFileName");

    // Load categories
    (async () => {
      try {
        const r = await apiGet("api/object_categories");
        state.categories = r.categories || [];
      } catch (_) {}
    })();

    // Autocomplete
    objectCategoryInput.addEventListener("input", () => {
      const q = (objectCategoryInput.value || "").trim().toLowerCase();
      categorySuggest.innerHTML = "";
      if (!q || !state.categories.length) return;
      state.categories.filter((c) => c.toLowerCase().includes(q)).slice(0, 10).forEach((cat) => {
        const li = document.createElement("li");
        li.textContent = cat;
        li.onclick = () => { objectCategoryInput.value = cat; categorySuggest.innerHTML = ""; };
        categorySuggest.appendChild(li);
      });
    });
    objectCategoryInput.addEventListener("blur", () => {
      setTimeout(() => { categorySuggest.innerHTML = ""; }, 150);
    });

    // Drag & drop
    function setDropFile(file) {
      const dt = new DataTransfer();
      if (file) dt.items.add(file);
      videoFileInput.files = dt.files;
      dropFileName.textContent = file ? file.name : "";
      dropFileName.style.display = file ? "block" : "none";
    }
    dropZone.addEventListener("click", () => videoFileInput.click());
    dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      const file = e.dataTransfer?.files?.[0];
      if (file && /\.(mp4|mkv|avi|mov|webm|flv)$/i.test(file.name)) setDropFile(file);
    });
    videoFileInput.addEventListener("change", () => setDropFile(videoFileInput.files?.[0] || null));

    // Process button
    $("processBtn").addEventListener("click", async () => {
      const objectCategory = (objectCategoryInput.value || "").trim();
      const file = videoFileInput.files?.[0];
      if (!objectCategory) { setStatus($("processStatus"), "Object category is required", "err"); return; }
      if (!file) { setStatus($("processStatus"), "Please upload a video file", "err"); return; }

      const formData = new FormData();
      formData.append("object_category", objectCategory);
      formData.append("video_file", file);

      $("processBtn").disabled = true;
      setStatus($("processStatus"), "Parsing... (scene detection, please wait)");
      $("segmentsPanel").style.display = "none";
      state.uploadToken = null;
      state.segments = [];
      state.selected.clear();

      try {
        const r = await apiPostForm("api/process", formData);
        if (!r.ok) throw new Error(r.error);
        state.uploadToken = r.token;
        state.segments = r.segments || [];
        setStatus($("processStatus"), `Done: ${state.segments.length} segments found`, "ok");
        renderSegments();
        $("segmentsPanel").style.display = "block";
      } catch (e) {
        setStatus($("processStatus"), "Failed: " + e.message, "err");
      }
      $("processBtn").disabled = false;
    });

    // Render segments
    function renderSegments() {
      const list = $("segmentsList");
      list.innerHTML = "";
      state.segments.forEach((seg, idx) => {
        const card = document.createElement("div");
        card.className = "segment-card" + (state.selected.has(idx) ? " selected" : "");
        card.innerHTML = `
          <div class="segment-video-wrap">
            <video class="segment-video" src="${seg.url || ""}" controls preload="metadata" muted playsinline></video>
          </div>
          <div class="idx">#${idx + 1}</div>
          <div class="time">${seg.start_str || seg.start} - ${seg.end_str || seg.end}</div>
          <div class="dur">${(seg.duration || 0).toFixed(1)}s</div>
        `;
        card.addEventListener("click", (e) => {
          if (!e.target.closest("video")) {
            if (state.selected.has(idx)) state.selected.delete(idx);
            else if (state.selected.size < state.maxSelect) state.selected.add(idx);
            renderSegments();
          }
        });
        list.appendChild(card);
      });
    }

    // Save segments
    $("saveSegBtn").addEventListener("click", async () => {
      if (!state.uploadToken || state.selected.size === 0) {
        setStatus($("saveStatus"), "Parse video and select at least 1 segment", "err");
        return;
      }
      $("saveSegBtn").disabled = true;
      setStatus($("saveStatus"), "Saving...");
      try {
        const r = await apiPost("api/save_segments", {
          token: state.uploadToken,
          selected: Array.from(state.selected).sort((a, b) => a - b),
          segments: state.segments,
        });
        if (!r.ok) throw new Error(r.error);
        setStatus($("saveStatus"), `Saved ${r.count} segment(s). Switch to Annotate tab to continue.`, "ok");
        state.uploadToken = null;
        state.segments = [];
        state.selected.clear();
        $("segmentsPanel").style.display = "none";
      } catch (e) {
        setStatus($("saveStatus"), "Failed: " + e.message, "err");
      }
      $("saveSegBtn").disabled = false;
    });
  }

  // ==================== Annotate Tab ====================
  function initAnnotateTab() {
    // Video time tracking
    const video = $("video");
    video.ontimeupdate = () => { state.cursorTimestamp = video.currentTime || 0; $("ts").textContent = state.cursorTimestamp.toFixed(3); };
    video.onseeked = () => { state.cursorTimestamp = video.currentTime || 0; $("ts").textContent = state.cursorTimestamp.toFixed(3); };

    $("refreshBtn").addEventListener("click", () => refreshSessions());
    $("searchQ").addEventListener("keydown", (e) => { if (e.key === "Enter") refreshSessions(); });

    $("captureBtn").addEventListener("click", () => captureFrame());
    $("selectIndexBtn").addEventListener("click", () => selectIndexFrame());
    $("ptypeHuman").addEventListener("click", () => setPointType("human"));
    $("ptypeObject").addEventListener("click", () => setPointType("object"));
    $("undoBtn").addEventListener("click", () => undoPoint());
    $("clearBtn").addEventListener("click", () => clearPoints());
    $("saveAnnoBtn").addEventListener("click", () => saveAnnotation().catch((e) => toast(String(e))));
    $("annotCanvas").addEventListener("click", canvasClickToPoint);
    window.addEventListener("resize", () => redrawOverlay());
  }

  async function refreshSessions() {
    const q = ($("searchQ").value || "").trim();
    try {
      const r = await apiGet(`api/sessions?q=${encodeURIComponent(q)}`);
      state.sessions = r.sessions || [];
      renderSessionList();
    } catch (e) {
      toast("Failed to load sessions: " + e.message);
    }
  }

  function renderSessionList() {
    const list = $("sessionList");
    list.innerHTML = "";
    state.sessions.forEach((s, i) => {
      const div = document.createElement("div");
      div.className = "session-item" + (i === state.sessionIdx ? " active" : "") + (s.annotated ? " done" : "");
      div.innerHTML = `
        <div class="k">${escapeHtml(s.category)}${s.category ? " / " : ""}${escapeHtml(s.name)}</div>
        <div class="badges">
          ${s.has_points ? '<span class="badge ok">points</span>' : ""}
          ${s.has_select_id ? '<span class="badge ok">select_id</span>' : ""}
          ${!s.annotated ? '<span class="badge warn">pending</span>' : '<span class="badge done">done</span>'}
        </div>
      `;
      div.onclick = () => loadSession(i);
      list.appendChild(div);
    });
  }

  async function loadSession(i) {
    if (i < 0 || i >= state.sessions.length) return;
    state.sessionIdx = i;
    const s = state.sessions[i];
    renderSessionList();

    $("annotPanel").style.display = "";
    $("curTitle").textContent = `${s.category}${s.category ? " / " : ""}${s.name}`;
    $("curMeta").textContent = s.path;

    const video = $("video");
    // Strip leading slash to avoid double-slash in URL; Flask <path:> handles the rest
    const cleanPath = s.path.replace(/^\//, "");
    video.src = `api/video/${cleanPath}`;
    video.load();

    // Reset annotation state
    state.points = { human: [], object: [] };
    state.annotFrameDataUrl = null;
    state.annotFrameSize = { w: 0, h: 0 };
    state.annotFrameTimestamp = null;
    state.selectedTimestamp = null;
    $("selTs").textContent = "\u2014";
    $("annotImg").removeAttribute("src");
    $("annotPlaceholder").style.display = "";
    redrawOverlay();
    setPointType("human");
  }

  function captureFrame() {
    if (state.sessionIdx < 0) return;
    const video = $("video");
    const canvas = $("hiddenCanvas");
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) { toast("Video metadata not loaded yet"); return; }

    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d").drawImage(video, 0, 0, w, h);

    const dataUrl = canvas.toDataURL("image/png");
    const ts = video.currentTime || 0;

    state.annotFrameDataUrl = dataUrl;
    state.annotFrameSize = { w, h };
    state.annotFrameTimestamp = ts;
    state.points = { human: [], object: [] };

    $("annotImg").src = dataUrl;
    $("annotPlaceholder").style.display = "none";
    redrawOverlay();
    toast(`Frame captured at t=${ts.toFixed(3)}s`);

    // Save capture to server
    const s = state.sessions[state.sessionIdx];
    apiPost("api/capture_frame", { session_path: s.path, image: dataUrl, timestamp: ts }).catch(() => {});
  }

  function selectIndexFrame() {
    const video = $("video");
    const t = video.currentTime || 0;
    if (state.annotFrameTimestamp != null && t < state.annotFrameTimestamp) {
      toast("Index frame must be >= captured frame time");
      return;
    }
    state.selectedTimestamp = t;
    $("selTs").textContent = t.toFixed(3);
    toast(`Index frame set at t=${t.toFixed(3)}s`);
  }

  function setPointType(t) {
    state.pointType = t;
    $("ptypeHuman").classList.toggle("active", t === "human");
    $("ptypeObject").classList.toggle("active", t === "object");
  }

  function getContainMapping(stageW, stageH, imgW, imgH) {
    if (!imgW || !imgH || !stageW || !stageH) return { scale: 1, offX: 0, offY: 0, dispW: stageW, dispH: stageH };
    const scale = Math.min(stageW / imgW, stageH / imgH);
    const dispW = imgW * scale, dispH = imgH * scale;
    return { scale, offX: (stageW - dispW) / 2, offY: (stageH - dispH) / 2, dispW, dispH };
  }

  function redrawOverlay() {
    const stage = $("annotStage");
    const canvas = $("annotCanvas");
    const ctx = canvas.getContext("2d");
    const rect = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!state.annotFrameDataUrl || !state.annotFrameSize.w) return;

    const { scale, offX, offY } = getContainMapping(rect.width, rect.height, state.annotFrameSize.w, state.annotFrameSize.h);
    const drawPts = (pts, color) => {
      ctx.fillStyle = color;
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 1.5 * dpr;
      for (const [x, y] of pts) {
        ctx.beginPath();
        ctx.arc((offX + x * scale) * dpr, (offY + y * scale) * dpr, 4.5 * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    };
    drawPts(state.points.human, "rgba(86,255,234,0.9)");
    drawPts(state.points.object, "rgba(122,95,255,0.9)");
  }

  function canvasClickToPoint(ev) {
    if (!state.annotFrameDataUrl) { toast("Capture a frame first"); return; }
    const stage = $("annotStage");
    const stageRect = stage.getBoundingClientRect();
    const { scale, offX, offY, dispW, dispH } = getContainMapping(
      stageRect.width, stageRect.height, state.annotFrameSize.w, state.annotFrameSize.h
    );
    const ix = (ev.clientX - stageRect.left) - offX;
    const iy = (ev.clientY - stageRect.top) - offY;
    if (ix < 0 || iy < 0 || ix > dispW || iy > dispH) return;

    const x = Math.round(ix / scale), y = Math.round(iy / scale);
    state.points[state.pointType].push([x, y]);
    redrawOverlay();
  }

  function undoPoint() {
    state.points[state.pointType].pop();
    redrawOverlay();
  }

  function clearPoints() {
    state.points = { human: [], object: [] };
    redrawOverlay();
  }

  async function saveAnnotation() {
    if (state.sessionIdx < 0) return;
    const s = state.sessions[state.sessionIdx];

    if (!state.annotFrameDataUrl) {
      window.alert("Please capture a frame first using the 'Capture Frame' button.");
      return;
    }
    if (state.selectedTimestamp === null) {
      window.alert("Please select an index frame:\n1) Seek video to desired frame\n2) Click 'Select Index Frame'\n3) Then click 'Save Annotation'");
      return;
    }
    if (state.selectedTimestamp < state.annotFrameTimestamp) {
      toast("Index frame must be >= captured frame time");
      return;
    }

    toast("Saving...");
    const resp = await apiPost("api/save_annotation", {
      session_path: s.path,
      human_points: state.points.human,
      object_points: state.points.object,
      start_timestamp: state.annotFrameTimestamp,
      selected_timestamp: state.selectedTimestamp,
      object_category: s.category,
    });
    toast(`Saved! select_id=${resp.select_id}, start_id=${resp.start_id}`);
    await refreshSessions();
  }

  // ==================== Init ====================
  function init() {
    // Tab switching
    document.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    initUploadTab();
    initAnnotateTab();
  }

  window.addEventListener("DOMContentLoaded", init);
})();
