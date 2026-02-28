"use strict";
(() => {
  // player.ts
  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }
  var chapterList = document.getElementById("chapterList");
  var currentChapterInfo = document.getElementById("currentChapterInfo");
  var timeDisplay = document.getElementById("timeDisplay");
  var chapterFilter = document.getElementById("chapterFilter");
  var playPauseBtn = document.getElementById("playPause");
  var player = null;
  var chapters = [];
  var queenKills = [];
  var currentChapterIndex = -1;
  var lastQueenKillIndex = -1;
  var timeUpdateInterval;
  var favoriteTeam = null;
  var selectedPosition = null;
  var playerHighlights = [];
  var playerHighlightCount = 0;
  var playerLowlightCount = 0;
  var lastHighlightIndex = -1;
  var users = {};
  var selectedUserId = null;
  var HIGHLIGHT_SEEK_BUFFER = 4.5;
  var GATE_Y_OFFSET = 4;
  var OVERLAY_LINE_HEIGHT = 20;
  var highlightModeEnabled = false;
  var HIGHLIGHT_PLAY_DURATION = 6;
  var POSITION_NAMES = {
    "1": "Gold Queen",
    "2": "Blue Queen",
    "3": "Gold Stripes",
    "4": "Gold Skull",
    "5": "Gold Abs",
    "6": "Gold Checkers",
    "7": "Blue Stripes",
    "8": "Blue Skull",
    "9": "Blue Abs",
    "10": "Blue Checkers"
  };
  var GOLD_COLOR = "#ffc107";
  var BLUE_COLOR = "#2196f3";
  var DARK_BG = "#1a1a1a";
  var SVG_TEMPLATES = {
    // Crown for queens
    crown: (color) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><path d="M5 16L3 5l5.5 5L12 4l3.5 6L21 5l-2 11H5zm14 3c0 .6-.4 1-1 1H6c-.6 0-1-.4-1-1v-1h14v1z"/></svg>`)}`,
    // Horizontal stripes
    stripes: (color) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><rect x="3" y="4" width="18" height="3" rx="1"/><rect x="3" y="10" width="18" height="3" rx="1"/><rect x="3" y="16" width="18" height="3" rx="1"/></svg>`)}`,
    // Skull
    skull: (color) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><path d="M12 2C6.5 2 2 6.5 2 12v3.5c0 1.4 1.1 2.5 2.5 2.5H6v-3h2v4h3v-4h2v4h3v-4h2v3h1.5c1.4 0 2.5-1.1 2.5-2.5V12c0-5.5-4.5-10-10-10zm-3 12a2 2 0 110-4 2 2 0 010 4zm6 0a2 2 0 110-4 2 2 0 010 4z"/></svg>`)}`,
    // Abs/muscular figure
    abs: (color) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><ellipse cx="12" cy="5" rx="4" ry="3"/><path d="M8 9h8v13H8V9z"/><rect x="9" y="10" width="2.5" height="3" fill="${DARK_BG}"/><rect x="12.5" y="10" width="2.5" height="3" fill="${DARK_BG}"/><rect x="9" y="14" width="2.5" height="3" fill="${DARK_BG}"/><rect x="12.5" y="14" width="2.5" height="3" fill="${DARK_BG}"/><rect x="9" y="18" width="2.5" height="2.5" fill="${DARK_BG}"/><rect x="12.5" y="18" width="2.5" height="2.5" fill="${DARK_BG}"/></svg>`)}`,
    // Checkerboard pattern
    checkers: (color) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect width="24" height="24" fill="${color}"/><rect x="0" y="0" width="6" height="6" fill="${DARK_BG}"/><rect x="12" y="0" width="6" height="6" fill="${DARK_BG}"/><rect x="6" y="6" width="6" height="6" fill="${DARK_BG}"/><rect x="18" y="6" width="6" height="6" fill="${DARK_BG}"/><rect x="0" y="12" width="6" height="6" fill="${DARK_BG}"/><rect x="12" y="12" width="6" height="6" fill="${DARK_BG}"/><rect x="6" y="18" width="6" height="6" fill="${DARK_BG}"/><rect x="18" y="18" width="6" height="6" fill="${DARK_BG}"/></svg>`)}`
  };
  var POSITION_SVGS = {
    "1": SVG_TEMPLATES.crown(GOLD_COLOR),
    // Gold Queen
    "2": SVG_TEMPLATES.crown(BLUE_COLOR),
    // Blue Queen
    "3": SVG_TEMPLATES.stripes(GOLD_COLOR),
    // Gold Stripes
    "4": SVG_TEMPLATES.skull(GOLD_COLOR),
    // Gold Skull
    "5": SVG_TEMPLATES.abs(GOLD_COLOR),
    // Gold Abs
    "6": SVG_TEMPLATES.checkers(GOLD_COLOR),
    // Gold Checkers
    "7": SVG_TEMPLATES.stripes(BLUE_COLOR),
    // Blue Stripes
    "8": SVG_TEMPLATES.skull(BLUE_COLOR),
    // Blue Skull
    "9": SVG_TEMPLATES.abs(BLUE_COLOR),
    // Blue Abs
    "10": SVG_TEMPLATES.checkers(BLUE_COLOR)
    // Blue Checkers
  };
  function getPositionIconImg(pos, size = 18) {
    if (!POSITION_SVGS[pos]) return "";
    return `<img src="${POSITION_SVGS[pos]}" width="${size}" height="${size}" style="vertical-align: middle; margin-right: 4px;" alt="${POSITION_NAMES[pos] || "Position"}">`;
  }
  var videoId = null;
  var chapterData = null;
  var youtubeApiReady = false;
  var videos = {};
  var currentVideoSource = null;
  var cabFilter = null;
  function initializePlayer() {
    if (!youtubeApiReady || !videoId || player) return;
    player = new YT.Player("player", {
      videoId,
      playerVars: {
        "autoplay": 0,
        "controls": 1,
        "rel": 0,
        "modestbranding": 1
      },
      events: {
        "onReady": onPlayerReady,
        "onStateChange": onPlayerStateChange
      }
    });
  }
  function onYouTubeIframeAPIReady() {
    youtubeApiReady = true;
    initializePlayer();
  }
  window.onYouTubeIframeAPIReady = onYouTubeIframeAPIReady;
  if (window.YT && window.YT.Player) {
    onYouTubeIframeAPIReady();
  }
  function onPlayerReady(event) {
    console.log("YouTube player ready");
    timeUpdateInterval = setInterval(updateCurrentChapter, 500);
    const params = new URLSearchParams(window.location.search);
    const gameParam = params.get("game");
    const tParam = params.get("t");
    if (gameParam) {
      const gid = parseInt(gameParam);
      const idx = chapters.findIndex((ch) => ch.game_id === gid);
      if (idx >= 0) {
        const seekTime = tParam ? chapters[idx].start_time + parseFloat(tParam) : chapters[idx].start_time;
        seekTo(seekTime);
        console.log(`URL nav: game ${gid} (chapter ${idx}), seeking to ${seekTime}s`);
      }
    } else if (tParam) {
      seekTo(parseFloat(tParam));
    }
  }
  function onPlayerStateChange(event) {
    if (event.data === YT.PlayerState.PLAYING) {
      playPauseBtn.textContent = "\u23F8 Pause";
    } else {
      playPauseBtn.textContent = "\u25B6 Play";
    }
  }
  function getCurrentTime() {
    return player && player.getCurrentTime ? player.getCurrentTime() : 0;
  }
  function getDuration() {
    return player && player.getDuration ? player.getDuration() : 0;
  }
  function seekTo(seconds) {
    if (player && player.seekTo) {
      player.seekTo(seconds, true);
    }
  }
  function togglePlayPause() {
    if (!player) return;
    const state = player.getPlayerState();
    if (state === YT.PlayerState.PLAYING) {
      player.pauseVideo();
    } else {
      player.playVideo();
    }
  }
  function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor(seconds % 3600 / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
    }
    return `${m}:${s.toString().padStart(2, "0")}`;
  }
  function findChapterAtTime(time) {
    for (let i = chapters.length - 1; i >= 0; i--) {
      if (time >= chapters[i].start_time) {
        if (currentVideoSource && chapters[i].video_source !== currentVideoSource) continue;
        return i;
      }
    }
    return -1;
  }
  function updateCurrentChapter() {
    const currentTime = getCurrentTime();
    const newIndex = findChapterAtTime(currentTime);
    if (newIndex !== currentChapterIndex) {
      currentChapterIndex = newIndex;
      if (selectedUserId && currentChapterIndex >= 0) {
        const pos = getUserPositionInChapter(selectedUserId, chapters[currentChapterIndex]);
        selectedPosition = pos ? String(pos) : null;
      }
      document.querySelectorAll(".chapter-item").forEach((el, i) => {
        el.classList.toggle("active", i === currentChapterIndex);
      });
      const activeEl = document.querySelector(".chapter-item.active");
      if (activeEl) {
        activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }
    if (currentChapterIndex >= 0) {
      const ch = chapters[currentChapterIndex];
      currentChapterInfo.innerHTML = `<h3>${esc(ch.title)}</h3>
            <span class="${esc(ch.winner)}">${esc(ch.winner)}</span> wins by ${esc(ch.win_condition)}
            &nbsp;|&nbsp; ${formatTime(ch.duration)}
            &nbsp;|&nbsp; <a href="${esc(ch.hivemind_url)}" target="_blank" style="color: #e94560;">HiveMind</a>`;
    }
    timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(getDuration())}`;
    checkHighlightAutoAdvance();
    updateContributionBars(currentTime);
    updateEggGrid(currentTime);
    updateBerryGrid(currentTime);
    updateOverlay(currentTime);
  }
  var CF_LABELS = {
    "bqk": ["Queen Kill", "blue"],
    "gqk": ["Queen Kill", "gold"],
    "bb": ["Berry", "blue"],
    "gb": ["Berry", "gold"],
    "bswd": ["Speed Warrior Dies", "blue"],
    "bvwd": ["Warrior Dies", "blue"],
    "gswd": ["Speed Warrior Dies", "gold"],
    "gvwd": ["Warrior Dies", "gold"],
    "bsdw": ["Speed Gets Wings", "blue"],
    "bdw": ["Gets Wings", "blue"],
    "bws": ["Gets Speed", "blue"],
    "gsdw": ["Speed Gets Wings", "gold"],
    "gdw": ["Gets Wings", "gold"],
    "gws": ["Gets Speed", "gold"],
    "sb": ["Snail \u2192 Blue", null],
    "sg": ["Snail \u2192 Gold", null]
  };
  for (let i = 0; i < 5; i++) {
    CF_LABELS[`mb${i}`] = [`Gate ${i} \u2192 Blue`, "blue"];
    CF_LABELS[`mg${i}`] = [`Gate ${i} \u2192 Gold`, "gold"];
  }
  var MAP_STRUCTURE = {
    "Day": {
      maiden_info: [
        ["maiden_speed", 410, 860],
        ["maiden_speed", 1510, 860],
        ["maiden_wings", 560, 260],
        ["maiden_wings", 960, 500],
        ["maiden_wings", 1360, 260]
      ],
      left_berries_centroid: [830, 937],
      right_berries_centroid: [1090, 937],
      snail_center: [960, 1010],
      blue_hive: [1860, 980],
      gold_hive: [60, 980],
      gold_eggs_centroid: [850, 899]
    },
    "Dusk": {
      maiden_info: [
        ["maiden_speed", 340, 140],
        ["maiden_speed", 1580, 140],
        ["maiden_wings", 310, 620],
        ["maiden_wings", 960, 140],
        ["maiden_wings", 1610, 620]
      ],
      left_berries_centroid: [800, 685],
      right_berries_centroid: [1120, 685],
      snail_center: [960, 870],
      blue_hive: [1860, 980],
      gold_hive: [60, 980],
      gold_eggs_centroid: [746, 532]
    },
    "Night": {
      maiden_info: [
        ["maiden_speed", 170, 740],
        ["maiden_speed", 1750, 740],
        ["maiden_wings", 700, 260],
        ["maiden_wings", 960, 700],
        ["maiden_wings", 1220, 260]
      ],
      left_berries_centroid: [170, 96],
      right_berries_centroid: [1750, 96],
      snail_center: [960, 970],
      blue_hive: [1860, 980],
      gold_hive: [60, 980],
      gold_eggs_centroid: [97, 55]
    },
    "Twilight": {
      maiden_info: [
        ["maiden_speed", 410, 860],
        ["maiden_speed", 1510, 860],
        ["maiden_wings", 550, 260],
        ["maiden_wings", 960, 820],
        ["maiden_wings", 1370, 260]
      ],
      left_berries_centroid: [158, 322],
      right_berries_centroid: [1762, 322],
      snail_center: [960, 1010],
      blue_hive: [1860, 980],
      gold_hive: [60, 980],
      gold_eggs_centroid: [164, 52]
    }
  };
  function getOverlayPosition(key, mapInfo, goldOnLeft, transform, snailX, cfDict) {
    if (!mapInfo) return null;
    function toPercent(x, y, flipX) {
      const px = flipX ? 1920 - x : x;
      const py = 1080 - y;
      if (transform) {
        return [
          transform.a_x + transform.b_x * px,
          transform.a_y + transform.b_y * py
        ];
      }
      return [px / 1920 * 100, py / 1080 * 100];
    }
    const needsFlip = !goldOnLeft;
    const maidenMatch = key.match(/^m[bg](\d)$/);
    if (maidenMatch) {
      const idx = parseInt(maidenMatch[1]);
      if (idx < mapInfo.maiden_info.length) {
        const [, mx, my] = mapInfo.maiden_info[idx];
        return [toPercent(mx, my, needsFlip)];
      }
      return null;
    }
    if (key === "bb") {
      const c = mapInfo.right_berries_centroid;
      return [toPercent(c[0], c[1], needsFlip)];
    }
    if (key === "gb") {
      const c = mapInfo.left_berries_centroid;
      return [toPercent(c[0], c[1], needsFlip)];
    }
    if (key === "sb" || key === "sg") {
      const sx = snailX != null ? snailX : mapInfo.snail_center[0];
      const sy = 1080 - mapInfo.snail_center[1];
      const yOff = key === "sb" ? OVERLAY_LINE_HEIGHT : -OVERLAY_LINE_HEIGHT;
      return [toPercent(sx, sy + yOff, false)];
    }
    if (key === "bqk" || key === "gqk") {
      const gc = mapInfo.gold_eggs_centroid || [850, 899];
      const blueX = 960 + (960 - gc[0]);
      const ey = gc[1] - 45;
      if (key === "gqk") {
        return [toPercent(goldOnLeft ? gc[0] : blueX, ey, false)];
      }
      return [toPercent(goldOnLeft ? blueX : gc[0], ey, false)];
    }
    if (key === "bvwd" || key === "bswd") {
      const h = mapInfo.blue_hive;
      const yOff = key === "bswd" ? OVERLAY_LINE_HEIGHT : 0;
      return [toPercent(h[0], h[1] + yOff, needsFlip)];
    }
    if (key === "gvwd" || key === "gswd") {
      const h = mapInfo.gold_hive;
      const yOff = key === "gswd" ? OVERLAY_LINE_HEIGHT : 0;
      return [toPercent(h[0], h[1] + yOff, needsFlip)];
    }
    if (key === "bdw" || key === "bsdw" || key === "gdw" || key === "gsdw") {
      const isBlue = key.startsWith("b");
      const yOff = isBlue ? -OVERLAY_LINE_HEIGHT : OVERLAY_LINE_HEIGHT;
      const positions = [];
      mapInfo.maiden_info.forEach(([type, mx, my], idx) => {
        if (type !== "maiden_wings") return;
        const flipKey = isBlue ? `mb${idx}` : `mg${idx}`;
        const isControlled = cfDict && !(flipKey in cfDict);
        if (isControlled) {
          positions.push(toPercent(mx, my + yOff, needsFlip));
        }
      });
      return positions.length > 0 ? positions : null;
    }
    if (key === "bws" || key === "gws") {
      const isBlue = key === "bws";
      const positions = [];
      mapInfo.maiden_info.forEach(([type, mx, my], idx) => {
        if (type !== "maiden_speed") return;
        const flipKey = isBlue ? `mb${idx}` : `mg${idx}`;
        const isControlled = cfDict && !(flipKey in cfDict);
        if (isControlled) {
          positions.push(toPercent(mx, my, needsFlip));
        }
      });
      return positions.length > 0 ? positions : null;
    }
    return null;
  }
  function updateOverlay(currentTime) {
    const overlay = document.getElementById("cfOverlay");
    if (!overlay) return;
    if (currentChapterIndex < 0) {
      overlay.innerHTML = "";
      return;
    }
    const ch = chapters[currentChapterIndex];
    if (!chapterData || !ch.model_timelines || !ch.map || ch.gold_on_left === void 0) {
      overlay.innerHTML = "";
      return;
    }
    const mapInfo = MAP_STRUCTURE[ch.map];
    if (!mapInfo) {
      overlay.innerHTML = "";
      return;
    }
    const point = findTimelinePoint(ch, currentTime, "c");
    if (!point || !point.c) {
      overlay.innerHTML = "";
      return;
    }
    const flipForGold = shouldFlipForGold();
    const positionedEntries = [];
    for (const [key, delta] of Object.entries(point.c)) {
      const displayDelta = flipForGold ? -delta : delta;
      if (Math.abs(displayDelta) < 5e-3) continue;
      const positions = getOverlayPosition(key, mapInfo, ch.gold_on_left, chapterData.game_transform, point.sx, point.c);
      if (!positions) continue;
      const isGate = /^m[bg]\d$/.test(key);
      for (const pos of positions) {
        positionedEntries.push({ key, delta: displayDelta, rawDelta: delta, x: pos[0], y: pos[1] + (isGate ? GATE_Y_OFFSET : 0) });
      }
    }
    if (positionedEntries.length === 0) {
      overlay.innerHTML = "";
      return;
    }
    const maxDelta = Math.max(0.1, ...positionedEntries.map((e) => Math.abs(e.delta)));
    const BLUE_CF = "rgba(59,130,246,0.9)";
    const ORANGE_CF = "rgba(249,115,22,0.9)";
    let html = "";
    for (const e of positionedEntries) {
      const fillPct = Math.abs(e.delta) / maxDelta * 50;
      const isBlueGood = e.rawDelta > 0;
      const color = isBlueGood ? BLUE_CF : ORANGE_CF;
      const goesRight = barGoesRight(e.rawDelta, e.delta, ch.gold_on_left);
      const barStyle = goesRight ? `left:50%;width:${fillPct}%;background:${color};` : `right:50%;width:${fillPct}%;background:${color};`;
      const pctText = `${e.key} ${(Math.abs(e.delta) * 100).toFixed(0)}%`;
      const labelColor = isBlueGood ? "#93c5fd" : "#fdba74";
      html += `<div class="cf-overlay-item" style="left:${e.x}%;top:${e.y}%;">
            <span class="cf-overlay-label" style="color:${labelColor}">${pctText}</span>
            <div class="cf-overlay-bar">
                <div class="cf-overlay-bar-fill" style="${barStyle}"></div>
            </div>
        </div>`;
    }
    overlay.innerHTML = html;
  }
  function findClosestPoint(timeline, time) {
    if (!timeline || timeline.length === 0) return null;
    let lo = 0, hi = timeline.length - 1;
    while (lo < hi) {
      const mid = lo + hi >> 1;
      if (timeline[mid].t < time) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0 && Math.abs(timeline[lo - 1].t - time) < Math.abs(timeline[lo].t - time)) {
      lo--;
    }
    return timeline[lo];
  }
  function findTimelinePoint(ch, currentTime, field) {
    if (!ch.model_timelines) return null;
    if (!ch._timelineFields) ch._timelineFields = {};
    for (const name of Object.keys(ch.model_timelines)) {
      const timeline = ch.model_timelines[name];
      if (!timeline || timeline.length === 0) continue;
      const cacheKey = name + ":" + field;
      if (!(cacheKey in ch._timelineFields)) {
        ch._timelineFields[cacheKey] = timeline.some((pt) => pt[field]);
      }
      if (!ch._timelineFields[cacheKey]) continue;
      return findClosestPoint(timeline, currentTime);
    }
    return null;
  }
  function updateContributionBars(currentTime) {
    const container = document.getElementById("contributionBars");
    const content = document.getElementById("cfBarsContent");
    if (currentChapterIndex < 0) {
      container.style.display = "none";
      return;
    }
    const ch = chapters[currentChapterIndex];
    if (!ch.model_timelines) {
      container.style.display = "none";
      return;
    }
    const point = findTimelinePoint(ch, currentTime, "c");
    if (!point || !point.c) {
      container.style.display = "none";
      return;
    }
    container.style.display = "";
    const flipForGold = shouldFlipForGold();
    const entries = [];
    for (const [key, delta] of Object.entries(point.c)) {
      const displayDelta = flipForGold ? -delta : delta;
      if (Math.abs(displayDelta) < 5e-3) continue;
      const labelInfo = CF_LABELS[key] || [key, null];
      entries.push({ key, label: `${key} ${labelInfo[0]}`, team: labelInfo[1], delta: displayDelta, rawDelta: delta });
    }
    entries.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
    const maxDelta = Math.max(0.1, ...entries.map((e) => Math.abs(e.delta)));
    let html = "";
    for (const e of entries) {
      const pct = Math.abs(e.delta) / maxDelta * 50;
      const barClass = e.rawDelta > 0 ? "positive" : "negative";
      const goesRight = barGoesRight(e.rawDelta, e.delta, ch.gold_on_left);
      const barStyle = goesRight ? `left: 50%; width: ${pct}%;` : `left: ${50 - pct}%; width: ${pct}%;`;
      const teamColor = e.team === "blue" ? "#5ba3ec" : e.team === "gold" ? "#ffd700" : "#aaa";
      const valueStr = (Math.abs(e.delta) * 100).toFixed(1) + "%";
      html += `<div class="cf-bar-row">
            <span class="cf-bar-label" style="color:${teamColor}">${e.label}</span>
            <div class="cf-bar-track">
                <div class="cf-bar-center"></div>
                <div class="cf-bar-fill ${barClass}" style="${barStyle}"></div>
            </div>
            <span class="cf-bar-value">${valueStr}</span>
        </div>`;
    }
    content.innerHTML = html;
  }
  function probToColor(prob) {
    const r = Math.round(249 + (59 - 249) * prob);
    const g = Math.round(115 + (130 - 115) * prob);
    const b = Math.round(22 + (246 - 22) * prob);
    return `rgba(${r},${g},${b},0.9)`;
  }
  function contourBorderCSS(probs, row, col, n, needsMirror) {
    const p = probs[row][col];
    if (p === null) return "";
    const side = p >= 0.5;
    function crosses(r, c) {
      if (r < 0 || r >= n || c < 0 || c >= n) return false;
      const np = probs[r][c];
      if (np === null) return false;
      return np >= 0.5 !== side;
    }
    const borderStyle = "2px solid rgba(255,255,255,0.85)";
    const parts = [];
    if (!needsMirror) {
      if (crosses(row - 1, col)) parts.push(`border-top:${borderStyle};`);
      if (crosses(row, col + 1)) parts.push(`border-right:${borderStyle};`);
      if (crosses(row + 1, col)) parts.push(`border-bottom:${borderStyle};`);
      if (crosses(row, col - 1)) parts.push(`border-left:${borderStyle};`);
    } else {
      if (crosses(row, col - 1)) parts.push(`border-top:${borderStyle};`);
      if (crosses(row + 1, col)) parts.push(`border-right:${borderStyle};`);
      if (crosses(row, col + 1)) parts.push(`border-bottom:${borderStyle};`);
      if (crosses(row - 1, col)) parts.push(`border-left:${borderStyle};`);
    }
    return parts.join("");
  }
  function renderDiamondGrid(opts) {
    const {
      probs,
      n,
      currentRow,
      currentCol,
      needsMirror,
      cellSize,
      fontSize,
      leftLabel,
      rightLabel,
      flipDisplay,
      leftEdgeLabels,
      rightEdgeLabels
    } = opts;
    const step = cellSize * Math.SQRT2 / 2;
    const halfCell = cellSize / 2;
    const diagSpan = 2 * n - 1;
    const titleHeight = 16;
    const containerWidth = diagSpan * step + cellSize + 20;
    const containerHeight = titleHeight + diagSpan * step + cellSize;
    const cx = containerWidth / 2;
    const topPad = halfCell + titleHeight;
    let html = `<div class="diamond-grid-container" style="width:${containerWidth}px;height:${containerHeight}px;">`;
    html += `<span class="diamond-axis-label" style="left:0;top:0;">${leftLabel}</span>`;
    html += `<span class="diamond-axis-label" style="right:0;top:0;">${rightLabel}</span>`;
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const prob = probs[row][col];
        if (prob === null || prob === void 0) continue;
        let dx = col - row;
        if (needsMirror) dx = -dx;
        const dy = col + row;
        const x = cx + dx * step - halfCell;
        const y = topPad + dy * step - halfCell;
        const pct = flipDisplay ? Math.round((1 - prob) * 100) : Math.round(prob * 100);
        const bgColor = probToColor(prob);
        const isCurrent = row === currentRow && col === currentCol;
        const currentClass = isCurrent ? " egg-current-bold" : "";
        const contour = contourBorderCSS(probs, row, col, n, needsMirror);
        html += `<div class="diamond-cell${currentClass}" style="left:${x}px;top:${y}px;width:${cellSize + 1}px;height:${cellSize + 1}px;background:${bgColor};${contour}"><span style="font-size:${fontSize}px;">${pct}</span></div>`;
      }
    }
    const tickGap = 4;
    const tickFontSize = Math.max(9, fontSize - 2);
    const perpDist = (halfCell + tickGap) / Math.SQRT2;
    for (let i = 0; i < n; i++) {
      const cellCY = topPad + i * step;
      const leftText = leftEdgeLabels ? leftEdgeLabels[i] : String(i);
      const rightText = rightEdgeLabels ? rightEdgeLabels[i] : String(i);
      if (leftText !== null) {
        const lx = cx - i * step - perpDist;
        const ly = cellCY - perpDist;
        html += `<span class="diamond-axis-label" style="right:${containerWidth - lx}px;top:${ly}px;transform:translateY(-50%);font-size:${tickFontSize}px;">${leftText}</span>`;
      }
      if (rightText !== null) {
        const rx = cx + i * step + perpDist;
        const ry = cellCY - perpDist;
        html += `<span class="diamond-axis-label" style="left:${rx}px;top:${ry}px;transform:translateY(-50%);font-size:${tickFontSize}px;">${rightText}</span>`;
      }
    }
    html += "</div>";
    return html;
  }
  function updateEggGrid(currentTime) {
    const container = document.getElementById("eggGrid");
    const content = document.getElementById("eggGridContent");
    if (currentChapterIndex < 0) {
      container.style.display = "none";
      return;
    }
    const ch = chapters[currentChapterIndex];
    if (!ch.model_timelines) {
      container.style.display = "none";
      return;
    }
    const point = findTimelinePoint(ch, currentTime, "eg");
    if (!point || !point.eg) {
      container.style.display = "none";
      return;
    }
    container.style.display = "";
    const flipForGold = shouldFlipForGold();
    const needsMirror = !!ch.gold_on_left;
    const eg = point.eg;
    const ee = point.ee;
    const n = 3;
    const eggProbs = [];
    let currentRow = -1, currentCol = -1;
    for (let row = 0; row < n; row++) {
      eggProbs[row] = [];
      for (let col = 0; col < n; col++) {
        const blueEggs = row;
        const goldEggs = col;
        const idx = blueEggs * n + goldEggs;
        const prob = eg[idx];
        eggProbs[row][col] = prob;
        if (ee && blueEggs === ee[0] && goldEggs === ee[1]) {
          currentRow = row;
          currentCol = col;
        }
      }
    }
    const leftTeam = ch.gold_on_left ? "Gold" : "Blue";
    const rightTeam = ch.gold_on_left ? "Blue" : "Gold";
    const leftLabel = `${leftTeam} eggs`;
    const rightLabel = `${rightTeam} eggs`;
    content.innerHTML = renderDiamondGrid({
      probs: eggProbs,
      n,
      currentRow,
      currentCol,
      needsMirror,
      cellSize: 66,
      fontSize: 15,
      leftLabel,
      rightLabel,
      flipDisplay: flipForGold
    });
  }
  var BERRY_DELTAS = [0, 1, 2, 3, 4];
  var MAX_FOOD = 12;
  function updateBerryGrid(currentTime) {
    const container = document.getElementById("berryGrid");
    const content = document.getElementById("berryGridContent");
    if (currentChapterIndex < 0) {
      container.style.display = "none";
      return;
    }
    const ch = chapters[currentChapterIndex];
    if (!ch.model_timelines) {
      container.style.display = "none";
      return;
    }
    const point = findTimelinePoint(ch, currentTime, "bg");
    if (!point || !point.bg) {
      container.style.display = "none";
      return;
    }
    container.style.display = "";
    const flipForGold = shouldFlipForGold();
    const needsMirror = !!ch.gold_on_left;
    const bg = point.bg;
    const n = BERRY_DELTAS.length;
    const bc = point.bc || [0, 0];
    const berryProbs = [];
    for (let row = 0; row < n; row++) {
      berryProbs[row] = [];
      for (let col = 0; col < n; col++) {
        const blueDelta = row;
        const goldDelta = col;
        if (bc[0] + blueDelta >= MAX_FOOD || bc[1] + goldDelta >= MAX_FOOD) {
          berryProbs[row][col] = null;
          continue;
        }
        const idx = blueDelta * n + goldDelta;
        const raw = bg[idx];
        if (raw === null || raw === void 0) {
          berryProbs[row][col] = null;
        } else {
          berryProbs[row][col] = raw;
        }
      }
    }
    const currentRow = 0, currentCol = 0;
    const leftTeam = ch.gold_on_left ? "Gold" : "Blue";
    const rightTeam = ch.gold_on_left ? "Blue" : "Gold";
    const leftLabel = `${leftTeam} berries left`;
    const rightLabel = `${rightTeam} berries left`;
    const berryLabel = (scored, d) => {
      const left = MAX_FOOD - scored - d;
      return left <= 0 ? null : String(left);
    };
    const blueLabels = BERRY_DELTAS.map((d) => berryLabel(bc[0], d));
    const goldLabels = BERRY_DELTAS.map((d) => berryLabel(bc[1], d));
    const leftEdgeLabels = ch.gold_on_left ? goldLabels : blueLabels;
    const rightEdgeLabels = ch.gold_on_left ? blueLabels : goldLabels;
    content.innerHTML = renderDiamondGrid({
      probs: berryProbs,
      n,
      currentRow,
      currentCol,
      needsMirror,
      cellSize: 40,
      fontSize: 11,
      leftLabel,
      rightLabel,
      flipDisplay: flipForGold,
      leftEdgeLabels,
      rightEdgeLabels
    });
  }
  function calculateNetWinProb(ch, positionId) {
    if (!positionId || !ch.player_events) return null;
    const pos = parseInt(positionId);
    let netDelta = 0;
    for (const evt of ch.player_events) {
      if (evt.positions && evt.positions.includes(pos)) {
        netDelta += evt.delta;
      }
    }
    return netDelta;
  }
  function findHighImpactRanges(ch, positionId) {
    if (!positionId || !ch.player_events) return [];
    const pos = parseInt(positionId);
    const playerEvents = ch.player_events.filter((evt) => evt.positions && evt.positions.includes(pos) && Math.abs(evt.delta) >= 0.05).sort((a, b) => a.time - b.time);
    if (playerEvents.length === 0) return [];
    const ranges = [];
    let rangeStart = null;
    let rangeEnd = null;
    let rangeDelta = 0;
    for (const evt of playerEvents) {
      if (rangeStart === null) {
        rangeStart = evt.time;
        rangeEnd = evt.time;
        rangeDelta = evt.delta;
      } else if (evt.time - rangeEnd <= 5) {
        rangeEnd = evt.time;
        rangeDelta += evt.delta;
      } else {
        if (Math.abs(rangeDelta) >= 0.1) {
          ranges.push({ start: rangeStart, end: rangeEnd, delta: rangeDelta });
        }
        rangeStart = evt.time;
        rangeEnd = evt.time;
        rangeDelta = evt.delta;
      }
    }
    if (rangeStart !== null && Math.abs(rangeDelta) >= 0.1) {
      ranges.push({ start: rangeStart, end: rangeEnd, delta: rangeDelta });
    }
    return ranges;
  }
  function calculateKD(ch, positionId) {
    if (!positionId || !ch.kill_events) return null;
    const pos = parseInt(positionId);
    let kills = 0;
    let deaths = 0;
    for (const evt of ch.kill_events) {
      if (evt.killer === pos) kills++;
      if (evt.victim === pos) deaths++;
    }
    return { kills, deaths };
  }
  function isGoldTeam(positionId) {
    const pos = parseInt(positionId);
    return pos % 2 === 1;
  }
  function getChapterPosition(ch) {
    if (!ch) ch = currentChapterIndex >= 0 ? chapters[currentChapterIndex] : null;
    if (!ch) return null;
    let pos = selectedPosition;
    if (selectedUserId) {
      const p = getUserPositionInChapter(selectedUserId, ch);
      pos = p ? String(p) : null;
    }
    return pos;
  }
  function shouldFlipForGold(ch) {
    if (favoriteTeam === "blue") return false;
    if (favoriteTeam === "gold") return true;
    const pos = getChapterPosition(ch);
    return !!pos && isGoldTeam(pos);
  }
  function perspectiveDelta(delta, position) {
    return position && isGoldTeam(String(position)) ? -delta : delta;
  }
  function barGoesRight(rawDelta, displayDelta, goldOnLeft) {
    return goldOnLeft !== void 0 ? rawDelta > 0 === !!goldOnLeft : displayDelta > 0;
  }
  function buildTimelinePath(timeline, startTime, duration, width, height, padding, flipForGold) {
    let pathD = "";
    for (let i = 0; i < timeline.length; i++) {
      const pt = timeline[i];
      const x = padding + (pt.t - startTime) / duration * (width - 2 * padding);
      const prob = flipForGold ? 1 - pt.p : pt.p;
      const y = height - padding - prob * (height - 2 * padding);
      if (i === 0) {
        pathD += `M ${x} ${y}`;
      } else {
        pathD += ` L ${x} ${y}`;
      }
    }
    return pathD;
  }
  var MODEL_COLORS = ["#e94560", "#5ba3ec", "#50c878", "#f5a623"];
  function renderWinProbPlot(ch, index) {
    const hasTimeline = ch.win_timeline && ch.win_timeline.length >= 2;
    const hasModels = ch.model_timelines && Object.keys(ch.model_timelines).length > 0;
    if (!hasTimeline && !hasModels) return "";
    const width = 280;
    const height = 36;
    const padding = 2;
    const startTime = ch.start_time;
    const endTime = ch.end_time;
    const duration = endTime - startTime;
    const chapterPosition = getChapterPosition(ch);
    const flipForGold = shouldFlipForGold(ch);
    const highImpactRanges = chapterPosition ? findHighImpactRanges(ch, chapterPosition) : [];
    const highlightsHtml = highImpactRanges.map((range) => {
      const x1 = padding + (range.start - startTime) / duration * (width - 2 * padding);
      const x2 = padding + (range.end - startTime) / duration * (width - 2 * padding);
      const rangeWidth = Math.max(x2 - x1, 6);
      const isGoodForPlayer = flipForGold ? range.delta < 0 : range.delta > 0;
      const color = isGoodForPlayer ? "rgba(76, 175, 80, 0.4)" : "rgba(244, 67, 54, 0.4)";
      return `<rect x="${x1 - 2}" y="0" width="${rangeWidth + 4}" height="${height}" fill="${color}"/>`;
    }).join("");
    let pathsHtml = "";
    let legendHtml = "";
    if (hasModels) {
      const modelNames = Object.keys(ch.model_timelines);
      if (hasTimeline) {
        const basePathD = buildTimelinePath(ch.win_timeline, startTime, duration, width, height, padding, flipForGold);
        pathsHtml += `<path d="${basePathD}" fill="none" stroke="#888" stroke-width="1" stroke-dasharray="3,2" opacity="0.6"/>`;
      }
      modelNames.forEach((name, idx) => {
        const timeline = ch.model_timelines[name];
        if (!timeline || timeline.length < 2) return;
        const color = MODEL_COLORS[idx % MODEL_COLORS.length];
        const pathD = buildTimelinePath(timeline, startTime, duration, width, height, padding, flipForGold);
        pathsHtml += `<path d="${pathD}" fill="none" stroke="${color}" stroke-width="1.5"/>`;
      });
      const legendItems = modelNames.map((name, idx) => {
        const color = MODEL_COLORS[idx % MODEL_COLORS.length];
        return `<span style="color:${color}; margin-right:8px; font-size:10px;">\u25CF ${name}</span>`;
      });
      if (hasTimeline) {
        legendItems.push('<span style="color:#888; font-size:10px;">\u2504 HiveMind</span>');
      }
      legendHtml = `<div style="display:flex; flex-wrap:wrap; gap:2px; margin-top:2px;">${legendItems.join("")}</div>`;
    } else {
      const pathD = buildTimelinePath(ch.win_timeline, startTime, duration, width, height, padding, flipForGold);
      pathsHtml = `<path d="${pathD}" fill="none" stroke="#888" stroke-width="1.5"/>`;
    }
    return `
        <div class="win-prob-plot" data-chapter="${index}">
            <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
                ${highlightsHtml}
                <line x1="${padding}" y1="${height / 2}" x2="${width - padding}" y2="${height / 2}"
                      stroke="#333" stroke-width="1" stroke-dasharray="2,2"/>
                ${pathsHtml}
            </svg>
            ${legendHtml}
        </div>
    `;
  }
  function renderChapters(filter = "") {
    const filterLower = filter.toLowerCase();
    const hasMultipleVideos = Object.keys(videos).length > 1;
    const cabFilterContainer = document.getElementById("cabFilter");
    if (cabFilterContainer) {
      if (hasMultipleVideos) {
        cabFilterContainer.style.display = "flex";
        cabFilterContainer.innerHTML = `<button class="cab-filter-btn ${cabFilter === null ? "active" : ""}" data-cab="">All</button>` + Object.entries(videos).map(
          ([key, vs]) => `<button class="cab-filter-btn ${cabFilter === key ? "active" : ""}" data-cab="${esc(key)}">${esc(vs.label)}</button>`
        ).join("");
        cabFilterContainer.querySelectorAll(".cab-filter-btn").forEach((btn) => {
          btn.addEventListener("click", () => {
            cabFilter = btn.dataset.cab || null;
            renderChapters(chapterFilter.value);
          });
        });
      } else {
        cabFilterContainer.style.display = "none";
      }
    }
    chapterList.innerHTML = chapters.map((ch, i) => {
      const chapterPosition = getChapterPosition(ch);
      if (selectedUserId && !chapterPosition) {
        return "";
      }
      if (cabFilter && ch.video_source !== cabFilter) {
        return "";
      }
      if (filter) {
        const searchText = `${ch.map} ${ch.winner} ${ch.win_condition} ${ch.game_id}`.toLowerCase();
        if (!searchText.includes(filterLower)) {
          return "";
        }
      }
      const winnerClass = ch.winner === "gold" ? "gold-win" : "blue-win";
      const activeClass = i === currentChapterIndex ? "active" : "";
      const setClass = ch.is_set_start ? "set-start" : "in-set";
      const setLabel = ch.is_set_start && ch.match_info ? `<div class="set-label"><span class="blue">${esc(ch.match_info.blue)}</span> vs <span class="gold">${esc(ch.match_info.gold)}</span></div>` : "";
      const plotHtml = renderWinProbPlot(ch, i);
      let cabBadgeHtml = "";
      if (hasMultipleVideos && ch.video_source && videos[ch.video_source]) {
        cabBadgeHtml = `<span class="cab-badge">${esc(videos[ch.video_source].label)}</span>`;
      }
      let statsHtml = "";
      if (chapterPosition) {
        const kd = calculateKD(ch, chapterPosition);
        const netProb = calculateNetWinProb(ch, chapterPosition);
        if (kd) {
          const playerNetProb = isGoldTeam(chapterPosition) ? -netProb : netProb;
          const netProbStr = playerNetProb !== null ? `<span class="${playerNetProb >= 0 ? "good-prob" : "bad-prob"}">${playerNetProb >= 0 ? "+" : ""}${(playerNetProb * 100).toFixed(0)}%</span>` : "";
          statsHtml = `<span class="kd-stats">${kd.kills}/${kd.deaths}</span> ${netProbStr}`;
        }
      }
      return `
                <div class="chapter-item ${winnerClass} ${activeClass} ${setClass}" data-index="${i}">
                    ${setLabel}
                    <div class="chapter-title">${cabBadgeHtml}${esc(ch.title)}</div>
                    <div class="chapter-meta">
                        <span class="winner ${esc(ch.winner)}">${esc(ch.winner)}</span> ${esc(ch.win_condition)}
                        &nbsp;|&nbsp; ${formatTime(ch.duration)}
                        ${statsHtml ? "&nbsp;|&nbsp;" + statsHtml : ""}
                    </div>
                    ${plotHtml}
                </div>
            `;
    }).join("");
    document.querySelectorAll(".chapter-item").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest(".win-prob-plot")) return;
        const index = parseInt(el.dataset.index);
        jumpToChapter(index);
      });
    });
    document.querySelectorAll(".win-prob-plot").forEach((plot) => {
      plot.addEventListener("click", (e) => {
        e.stopPropagation();
        const chapterIndex = parseInt(plot.dataset.chapter);
        const ch = chapters[chapterIndex];
        const rect = plot.getBoundingClientRect();
        const clickX = (e.clientX - rect.left) / rect.width;
        const targetTime = ch.start_time + clickX * (ch.end_time - ch.start_time);
        seekTo(targetTime);
        if (player) player.playVideo();
      });
    });
  }
  function jumpToChapter(index) {
    if (index >= 0 && index < chapters.length) {
      const ch = chapters[index];
      const targetTime = ch.start_time;
      console.log(`Jumping to chapter ${index}: ${ch.title} at ${targetTime}s`);
      if (ch.video_source && videos[ch.video_source]) {
        const targetVideoId = videos[ch.video_source].video_id;
        if (player && targetVideoId !== videoId) {
          currentVideoSource = ch.video_source;
          videoId = targetVideoId;
          player.loadVideoById(targetVideoId, targetTime);
          currentChapterIndex = index;
          updateCurrentChapter();
          return;
        }
      }
      seekTo(targetTime);
      currentChapterIndex = index;
      updateCurrentChapter();
      if (player) player.playVideo();
    }
  }
  function prevChapter() {
    const idx = findChapterAtTime(getCurrentTime());
    if (idx > 0) {
      jumpToChapter(idx - 1);
    } else if (idx === 0) {
      jumpToChapter(0);
    }
  }
  function nextChapter() {
    const idx = findChapterAtTime(getCurrentTime());
    if (idx < chapters.length - 1) {
      jumpToChapter(idx + 1);
    } else if (idx === -1 && chapters.length > 0) {
      jumpToChapter(0);
    }
  }
  function nextSet() {
    const idx = findChapterAtTime(getCurrentTime());
    for (let i = idx + 1; i < chapters.length; i++) {
      if (chapters[i].is_set_start) {
        jumpToChapter(i);
        return;
      }
    }
  }
  function prevSet() {
    const idx = findChapterAtTime(getCurrentTime());
    const currentSetStart = chapters.findIndex(
      (ch, i) => i <= idx && ch.is_set_start && (i === idx || !chapters.slice(i + 1, idx + 1).some((c) => c.is_set_start))
    );
    for (let i = idx - 1; i >= 0; i--) {
      if (chapters[i].is_set_start) {
        if (i === currentSetStart && getCurrentTime() - chapters[i].start_time < 2) {
          continue;
        }
        jumpToChapter(i);
        return;
      }
    }
    if (chapters.length > 0) {
      jumpToChapter(0);
    }
  }
  function findQueenKillIndexAtTime(time) {
    for (let i = queenKills.length - 1; i >= 0; i--) {
      if (queenKills[i].time <= time + 2) {
        return i;
      }
    }
    return -1;
  }
  function nextQueenKill() {
    if (queenKills.length === 0) return;
    const currentTime = getCurrentTime();
    let nextIndex;
    if (lastQueenKillIndex >= 0 && lastQueenKillIndex < queenKills.length - 1) {
      const lastKillTime = queenKills[lastQueenKillIndex].time;
      if (Math.abs(currentTime - (lastKillTime - 1)) < 3) {
        nextIndex = lastQueenKillIndex + 1;
      }
    }
    if (nextIndex === void 0) {
      nextIndex = findQueenKillIndexAtTime(currentTime) + 1;
    }
    if (nextIndex < queenKills.length) {
      lastQueenKillIndex = nextIndex;
      seekTo(queenKills[nextIndex].time - 1);
      if (player) player.playVideo();
    }
  }
  function prevQueenKill() {
    if (queenKills.length === 0) return;
    const currentTime = getCurrentTime();
    if (lastQueenKillIndex >= 0) {
      const lastKillTime = queenKills[lastQueenKillIndex].time;
      const targetTime = lastKillTime - 1;
      if (currentTime > lastKillTime - 0.5) {
        seekTo(targetTime);
        if (player) player.playVideo();
        return;
      }
      if (Math.abs(currentTime - targetTime) < 2 && lastQueenKillIndex > 0) {
        lastQueenKillIndex = lastQueenKillIndex - 1;
        seekTo(queenKills[lastQueenKillIndex].time - 1);
        if (player) player.playVideo();
        return;
      }
    }
    const idx = findQueenKillIndexAtTime(currentTime);
    if (idx >= 0) {
      lastQueenKillIndex = idx;
      seekTo(queenKills[idx].time - 1);
      if (player) player.playVideo();
    }
  }
  function updatePlayerHighlights() {
    playerHighlights = [];
    lastHighlightIndex = -1;
    let allEvents = [];
    let anyQueen = false;
    const noSelection = !selectedUserId && !selectedPosition;
    for (const ch of chapters) {
      if (!ch.player_events) continue;
      if (noSelection) {
        for (const evt of ch.player_events) {
          if (evt.ml_score !== void 0 || Math.abs(evt.delta) >= 0.1) {
            allEvents.push({
              time: evt.time,
              delta: evt.delta,
              type: evt.type,
              game_id: ch.game_id,
              set_number: ch.set_number,
              event_id: evt.id,
              values: evt.values,
              position: evt.positions ? evt.positions[0] : null,
              ml_score: evt.ml_score
            });
          }
        }
      } else {
        let pos;
        if (selectedUserId) {
          pos = getUserPositionInChapter(selectedUserId, ch);
          if (!pos) continue;
        } else {
          pos = parseInt(selectedPosition);
        }
        if (pos === 1 || pos === 2) anyQueen = true;
        for (const evt of ch.player_events) {
          if (evt.positions && evt.positions.includes(pos)) {
            allEvents.push({
              time: evt.time,
              delta: evt.delta,
              type: evt.type,
              game_id: ch.game_id,
              set_number: ch.set_number,
              event_id: evt.id,
              values: evt.values,
              position: pos
            });
          }
        }
      }
    }
    const baseThreshold = anyQueen ? 0.2 : 0.15;
    allEvents.sort((a, b) => a.time - b.time);
    const windowSize = 5;
    for (let i = 0; i < allEvents.length; i++) {
      const evt = allEvents[i];
      if (evt.ml_score !== void 0) {
        evt.score = (evt.ml_score - 1) / 4;
      } else {
        let clusterScore = Math.abs(evt.delta);
        for (let j = 0; j < allEvents.length; j++) {
          if (i !== j && Math.abs(evt.time - allEvents[j].time) < windowSize) {
            clusterScore += Math.abs(allEvents[j].delta) * 0.3;
          }
        }
        evt.score = clusterScore;
      }
    }
    const targetPerSet = 4;
    const eventsBySet = {};
    for (const evt of allEvents) {
      const threshold = evt.ml_score !== void 0 ? 0.05 : baseThreshold;
      if (evt.score >= threshold) {
        if (!eventsBySet[evt.set_number]) {
          eventsBySet[evt.set_number] = [];
        }
        eventsBySet[evt.set_number].push(evt);
      }
    }
    const MIN_HIGHLIGHT_GAP = 10;
    for (const setNum in eventsBySet) {
      eventsBySet[setNum].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
      const selected = [];
      for (const evt of eventsBySet[setNum]) {
        const tooClose = selected.some((s) => Math.abs(s.time - evt.time) < MIN_HIGHLIGHT_GAP);
        if (!tooClose) {
          selected.push(evt);
          if (selected.length >= targetPerSet) break;
        }
      }
      playerHighlights.push(...selected);
    }
    const hasPositiveHighlight = playerHighlights.some((h) => {
      return perspectiveDelta(h.delta, h.position) > 0;
    });
    if (!hasPositiveHighlight && allEvents.length > 0) {
      let bestPositiveMove = null;
      let bestPositiveDelta = 0;
      for (const evt of allEvents) {
        const displayDelta = perspectiveDelta(evt.delta, evt.position);
        if (displayDelta > bestPositiveDelta) {
          bestPositiveDelta = displayDelta;
          bestPositiveMove = evt;
        }
      }
      if (bestPositiveMove && !playerHighlights.some((h) => h.id === bestPositiveMove.id)) {
        playerHighlights.push(bestPositiveMove);
      }
    }
    playerHighlights.sort((a, b) => a.time - b.time);
    playerHighlightCount = 0;
    playerLowlightCount = 0;
    for (const h of playerHighlights) {
      if (perspectiveDelta(h.delta, h.position) >= 0) playerHighlightCount++;
      else playerLowlightCount++;
    }
    document.getElementById("highlightCount").innerHTML = `<span class="good-prob">${playerHighlightCount}</span> / <span class="bad-prob">${playerLowlightCount}</span>`;
    renderHighlightDebug();
  }
  function renderHighlightDebug() {
    const debugEl = document.getElementById("highlightDebug");
    if (!selectedUserId && !selectedPosition) {
      debugEl.innerHTML = "";
      return;
    }
    if (playerHighlights.length === 0) {
      debugEl.innerHTML = '<p class="highlight-debug-empty">No highlights found for this player</p>';
      return;
    }
    let playerName = "Selected Player";
    let playerIcon = "";
    if (selectedUserId && users[selectedUserId]) {
      playerName = users[selectedUserId].name;
      if (currentChapterIndex >= 0) {
        const pos = getUserPositionInChapter(selectedUserId, chapters[currentChapterIndex]);
        if (pos) playerIcon = getPositionIconImg(String(pos), 20);
      }
    } else if (selectedPosition) {
      playerName = POSITION_NAMES[selectedPosition] || `Position ${selectedPosition}`;
      playerIcon = getPositionIconImg(selectedPosition, 20);
    }
    const itemsHtml = playerHighlights.map((h, idx) => {
      const displayDelta = perspectiveDelta(h.delta, h.position);
      const deltaClass = displayDelta >= 0 ? "positive" : "negative";
      const deltaStr = (displayDelta >= 0 ? "+" : "") + (displayDelta * 100).toFixed(0) + "%";
      const scoreStr = h.score ? `(${(h.score * 100).toFixed(0)})` : "";
      const valuesStr = h.values ? h.values.join(", ") : "";
      const eventIdStr = h.event_id ? `#${h.event_id}` : "";
      const posIcon = h.position ? getPositionIconImg(String(h.position), 14) : "";
      return `
            <div class="highlight-debug-item" data-highlight-index="${idx}">
                <span class="highlight-debug-pos">${posIcon}</span>
                <span class="highlight-debug-time">${formatTime(h.time)}</span>
                <span class="highlight-debug-delta ${deltaClass}">${deltaStr}</span>
                <span class="highlight-debug-type">${h.type || "event"} ${scoreStr}</span>
                <span class="highlight-debug-game">Game ${h.game_id} ${eventIdStr}</span>
            </div>
            ${valuesStr ? `<div class="highlight-debug-values">[${valuesStr}]</div>` : ""}
        `;
    }).join("");
    debugEl.innerHTML = `
        <h4>${playerIcon}${esc(playerName)} - <span class="good-prob">${playerHighlightCount}</span> / <span class="bad-prob">${playerLowlightCount}</span></h4>
        ${itemsHtml}
    `;
    debugEl.querySelectorAll(".highlight-debug-item").forEach((el) => {
      el.addEventListener("click", () => {
        const idx = parseInt(el.dataset.highlightIndex);
        lastHighlightIndex = idx;
        updateDebugActiveHighlight();
        seekTo(playerHighlights[idx].time - HIGHLIGHT_SEEK_BUFFER);
        if (player) player.playVideo();
      });
    });
  }
  function updateDebugActiveHighlight() {
    const debugEl = document.getElementById("highlightDebug");
    debugEl.querySelectorAll(".highlight-debug-item").forEach((el) => {
      const idx = parseInt(el.dataset.highlightIndex);
      el.classList.toggle("active", idx === lastHighlightIndex);
    });
    const activeEl = debugEl.querySelector(".highlight-debug-item.active");
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }
  function nextHighlight() {
    if (playerHighlights.length === 0) return;
    const currentTime = getCurrentTime();
    let nextIndex;
    if (lastHighlightIndex >= 0 && lastHighlightIndex < playerHighlights.length - 1) {
      const lastTime = playerHighlights[lastHighlightIndex].time;
      if (Math.abs(currentTime - (lastTime - HIGHLIGHT_SEEK_BUFFER)) < 3) {
        nextIndex = lastHighlightIndex + 1;
      }
    }
    if (nextIndex === void 0) {
      for (let i = 0; i < playerHighlights.length; i++) {
        if (playerHighlights[i].time > currentTime + 0.5) {
          nextIndex = i;
          break;
        }
      }
    }
    if (nextIndex !== void 0 && nextIndex < playerHighlights.length) {
      lastHighlightIndex = nextIndex;
      updateDebugActiveHighlight();
      seekTo(playerHighlights[nextIndex].time - HIGHLIGHT_SEEK_BUFFER);
      if (player) player.playVideo();
    }
  }
  function prevHighlight() {
    if (playerHighlights.length === 0) return;
    const currentTime = getCurrentTime();
    if (lastHighlightIndex >= 0) {
      const lastTime = playerHighlights[lastHighlightIndex].time;
      if (currentTime > lastTime - 0.5) {
        seekTo(lastTime - HIGHLIGHT_SEEK_BUFFER);
        if (player) player.playVideo();
        return;
      }
      if (Math.abs(currentTime - (lastTime - HIGHLIGHT_SEEK_BUFFER)) < 2 && lastHighlightIndex > 0) {
        lastHighlightIndex--;
        updateDebugActiveHighlight();
        seekTo(playerHighlights[lastHighlightIndex].time - HIGHLIGHT_SEEK_BUFFER);
        if (player) player.playVideo();
        return;
      }
    }
    for (let i = playerHighlights.length - 1; i >= 0; i--) {
      if (playerHighlights[i].time < currentTime - 1) {
        lastHighlightIndex = i;
        updateDebugActiveHighlight();
        seekTo(playerHighlights[i].time - HIGHLIGHT_SEEK_BUFFER);
        if (player) player.playVideo();
        return;
      }
    }
  }
  function toggleHighlightMode() {
    highlightModeEnabled = !highlightModeEnabled;
    const btn = document.getElementById("highlightModeBtn");
    const mBtn = document.getElementById("mHighlightMode");
    btn.classList.toggle("highlight-mode-active", highlightModeEnabled);
    mBtn.classList.toggle("highlight-mode-active", highlightModeEnabled);
    btn.textContent = highlightModeEnabled ? "Stop" : "Auto HL";
    mBtn.textContent = highlightModeEnabled ? "Stop" : "Auto";
    if (highlightModeEnabled) {
      if (playerHighlights.length === 0) {
        highlightModeEnabled = false;
        btn.classList.remove("highlight-mode-active");
        mBtn.classList.remove("highlight-mode-active");
        btn.textContent = "Auto HL";
        mBtn.textContent = "Auto";
        return;
      }
      const currentTime = getCurrentTime();
      let startIdx = 0;
      for (let i = 0; i < playerHighlights.length; i++) {
        if (playerHighlights[i].time > currentTime) {
          startIdx = i;
          break;
        }
      }
      lastHighlightIndex = startIdx;
      updateDebugActiveHighlight();
      seekTo(playerHighlights[startIdx].time - HIGHLIGHT_SEEK_BUFFER);
      if (player) player.playVideo();
    }
  }
  function checkHighlightAutoAdvance() {
    if (!highlightModeEnabled || playerHighlights.length === 0) return;
    if (lastHighlightIndex < 0) return;
    const currentTime = getCurrentTime();
    const currentHighlight = playerHighlights[lastHighlightIndex];
    if (currentTime >= currentHighlight.time + HIGHLIGHT_PLAY_DURATION) {
      if (lastHighlightIndex < playerHighlights.length - 1) {
        lastHighlightIndex++;
        updateDebugActiveHighlight();
        seekTo(playerHighlights[lastHighlightIndex].time - HIGHLIGHT_SEEK_BUFFER);
      } else {
        toggleHighlightMode();
      }
    }
  }
  function loadChaptersFromJSON(data) {
    chapterData = data;
    chapters = data.chapters || [];
    users = data.users || {};
    videoId = data.video_id || null;
    videos = data.videos || {};
    const videoKeys = Object.keys(videos);
    if (videoKeys.length > 0 && !videoId) {
      videoId = videos[videoKeys[0]].video_id;
      currentVideoSource = videoKeys[0];
    } else if (videoKeys.length > 0) {
      for (const key of videoKeys) {
        if (videos[key].video_id === videoId) {
          currentVideoSource = key;
          break;
        }
      }
    }
    queenKills = [];
    for (const ch of chapters) {
      if (ch.queen_kills) {
        for (const qk of ch.queen_kills) {
          queenKills.push({
            time: qk.time,
            victim: qk.victim,
            game_id: ch.game_id
          });
        }
      }
    }
    queenKills.sort((a, b) => a.time - b.time);
    const playerSelect = document.getElementById("playerSelect");
    const mobilePlayerSelect = document.getElementById("mobilePlayerSelect");
    const sortedUsers = Object.entries(users).sort((a, b) => a[1].name.toLowerCase().localeCompare(b[1].name.toLowerCase()));
    if (sortedUsers.length === 0) {
      const noUsersMsg = document.createElement("span");
      noUsersMsg.style.color = "#888";
      noUsersMsg.style.fontStyle = "italic";
      noUsersMsg.textContent = "No logged in users";
      playerSelect.parentNode.replaceChild(noUsersMsg, playerSelect);
      mobilePlayerSelect.style.display = "none";
    } else {
      playerSelect.innerHTML = '<option value="">Select player...</option>';
      mobilePlayerSelect.innerHTML = '<option value="">Player...</option>';
      for (const [userId, userInfo] of sortedUsers) {
        const option = document.createElement("option");
        option.value = userId;
        option.textContent = userInfo.name;
        playerSelect.appendChild(option);
        mobilePlayerSelect.appendChild(option.cloneNode(true));
      }
      const urlParams2 = new URLSearchParams(window.location.search);
      const playerParam = urlParams2.get("player");
      if (playerParam && users[playerParam]) {
        playerSelect.value = playerParam;
        mobilePlayerSelect.value = playerParam;
        handlePlayerSelect(playerParam, false);
      }
    }
    renderChapters();
    updateCurrentChapter();
    console.log(`Loaded ${chapters.length} chapters with ${queenKills.length} queen kills and ${Object.keys(users).length} users`);
    if (chapters.length > 0) {
      console.log(`First chapter at ${chapters[0].start_time}s, last at ${chapters[chapters.length - 1].start_time}s`);
    }
    updatePlayerHighlights();
    initializePlayer();
  }
  function cycleTeamToggle() {
    if (favoriteTeam === null) favoriteTeam = "blue";
    else if (favoriteTeam === "blue") favoriteTeam = "gold";
    else favoriteTeam = null;
    const btn = document.getElementById("teamToggle");
    if (favoriteTeam === null) btn.textContent = "Team: Auto";
    else if (favoriteTeam === "blue") btn.textContent = "Team: Blue";
    else btn.textContent = "Team: Gold";
    renderChapters(chapterFilter.value);
    if (player && player.getCurrentTime) {
      const t = player.getCurrentTime();
      updateContributionBars(t);
      updateEggGrid(t);
      updateBerryGrid(t);
      updateOverlay(t);
    }
  }
  document.getElementById("prevChapter").addEventListener("click", prevChapter);
  document.getElementById("nextChapter").addEventListener("click", nextChapter);
  document.getElementById("prevSet").addEventListener("click", prevSet);
  document.getElementById("nextSet").addEventListener("click", nextSet);
  document.getElementById("prevEgg").addEventListener("click", prevQueenKill);
  document.getElementById("nextEgg").addEventListener("click", nextQueenKill);
  document.getElementById("prevHighlightBtn").addEventListener("click", prevHighlight);
  document.getElementById("nextHighlightBtn").addEventListener("click", nextHighlight);
  document.getElementById("highlightModeBtn").addEventListener("click", toggleHighlightMode);
  document.getElementById("teamToggle").addEventListener("click", cycleTeamToggle);
  playPauseBtn.addEventListener("click", togglePlayPause);
  chapterFilter.addEventListener("input", (e) => {
    renderChapters(e.target.value);
  });
  function getUserPositionInChapter(userId, chapter) {
    if (!chapter.users) return null;
    for (const [pos, uid] of Object.entries(chapter.users)) {
      if (String(uid) === String(userId)) {
        return parseInt(pos);
      }
    }
    return null;
  }
  function handlePlayerSelect(userId, updateUrl = true) {
    selectedUserId = userId;
    favoriteTeam = null;
    document.getElementById("teamToggle").textContent = "Team: Auto";
    document.getElementById("positionSelect").value = "";
    document.getElementById("mobilePositionSelect").value = "";
    if (selectedUserId && currentChapterIndex >= 0) {
      const pos = getUserPositionInChapter(selectedUserId, chapters[currentChapterIndex]);
      selectedPosition = pos ? String(pos) : null;
    } else {
      selectedPosition = null;
    }
    if (updateUrl) {
      const url = new URL(window.location.href);
      if (userId) {
        url.searchParams.set("player", userId);
      } else {
        url.searchParams.delete("player");
      }
      window.history.replaceState({}, "", url);
    }
    updatePlayerHighlights();
    renderChapters(chapterFilter.value);
  }
  document.getElementById("playerSelect").addEventListener("change", (e) => {
    handlePlayerSelect(e.target.value);
    document.getElementById("mobilePlayerSelect").value = e.target.value;
  });
  document.getElementById("mobilePlayerSelect").addEventListener("change", (e) => {
    handlePlayerSelect(e.target.value);
    document.getElementById("playerSelect").value = e.target.value;
  });
  function handlePositionSelect(position) {
    selectedPosition = position;
    selectedUserId = null;
    favoriteTeam = null;
    document.getElementById("teamToggle").textContent = "Team: Auto";
    document.getElementById("playerSelect").value = "";
    document.getElementById("mobilePlayerSelect").value = "";
    document.getElementById("positionSelect").value = position;
    document.getElementById("mobilePositionSelect").value = position;
    updatePlayerHighlights();
    renderChapters(chapterFilter.value);
  }
  document.getElementById("positionSelect").addEventListener("change", (e) => {
    handlePositionSelect(e.target.value);
  });
  var mobileControlsVisible = true;
  document.getElementById("mobileToggle").addEventListener("click", () => {
    mobileControlsVisible = !mobileControlsVisible;
    document.getElementById("mobileControls").classList.toggle("visible", mobileControlsVisible);
    document.getElementById("mobileToggle").textContent = mobileControlsVisible ? "\u2715" : "\u2630";
  });
  document.getElementById("mPrevSet").addEventListener("click", prevSet);
  document.getElementById("mNextSet").addEventListener("click", nextSet);
  document.getElementById("mPrevGame").addEventListener("click", prevChapter);
  document.getElementById("mNextGame").addEventListener("click", nextChapter);
  document.getElementById("mPrevHighlight").addEventListener("click", prevHighlight);
  document.getElementById("mNextHighlight").addEventListener("click", nextHighlight);
  document.getElementById("mHighlightMode").addEventListener("click", toggleHighlightMode);
  document.getElementById("mobilePositionSelect").addEventListener("change", (e) => {
    handlePositionSelect(e.target.value);
  });
  document.getElementById("chaptersInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const data = JSON.parse(event.target.result);
          loadChaptersFromJSON(data);
        } catch (err) {
          alert("Error parsing chapters.json: " + err.message);
        }
      };
      reader.readAsText(file);
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    switch (e.key) {
      case " ":
        e.preventDefault();
        togglePlayPause();
        break;
      case "ArrowLeft":
        seekTo(getCurrentTime() - 5);
        break;
      case "ArrowRight":
        seekTo(getCurrentTime() + 5);
        break;
      case "j":
        seekTo(getCurrentTime() - 10);
        break;
      case "l":
        seekTo(getCurrentTime() + 10);
        break;
      case "p":
        prevChapter();
        break;
      case "n":
        nextChapter();
        break;
      case "g":
        nextChapter();
        break;
      case "G":
        prevChapter();
        break;
      case "s":
        nextSet();
        break;
      case "S":
        prevSet();
        break;
      case "e":
        nextQueenKill();
        break;
      case "E":
        prevQueenKill();
        break;
      case "h":
        nextHighlight();
        break;
      case "H":
        prevHighlight();
        break;
      case "a":
      case "A":
        toggleHighlightMode();
        break;
      case "t":
      case "T":
        cycleTeamToggle();
        break;
      case "1":
      case "2":
      case "3":
      case "4":
      case "5":
      case "6":
      case "7":
      case "8":
      case "9":
      case "0": {
        const keyToPosition = {
          "1": "10",
          // Blue Checkers
          "2": "8",
          // Blue Skull
          "3": "2",
          // Blue Queen
          "4": "9",
          // Blue Abs
          "5": "7",
          // Blue Stripes
          "6": "6",
          // Gold Checkers
          "7": "4",
          // Gold Skull
          "8": "1",
          // Gold Queen
          "9": "5",
          // Gold Abs
          "0": "3"
          // Gold Stripes
        };
        handlePositionSelect(keyToPosition[e.key]);
        break;
      }
    }
  });
  var calibrationClicks = [];
  var calibrating = false;
  var calibrationGates = null;
  function getSpeedGates(ch) {
    if (!ch || !ch.map) return null;
    const mapInfo = MAP_STRUCTURE[ch.map];
    if (!mapInfo) return null;
    const speedGates = mapInfo.maiden_info.filter((m) => m[0] === "maiden_speed").map((m) => ({ gx: m[1], gy: m[2] }));
    if (speedGates.length < 2) return null;
    const needsFlip = !ch.gold_on_left;
    const effectiveGates = speedGates.map((g) => ({
      gx: needsFlip ? 1920 - g.gx : g.gx,
      gy: g.gy
    }));
    effectiveGates.sort((a, b) => a.gx - b.gx);
    return { left: effectiveGates[0], right: effectiveGates[effectiveGates.length - 1] };
  }
  document.getElementById("calibrateBtn").addEventListener("click", () => {
    calibrating = !calibrating;
    calibrationClicks = [];
    calibrationGates = null;
    const btn = document.getElementById("calibrateBtn");
    const overlay = document.getElementById("cfOverlay");
    if (calibrating) {
      const ch = currentChapterIndex >= 0 ? chapters[currentChapterIndex] : null;
      calibrationGates = getSpeedGates(ch);
      if (!calibrationGates) {
        btn.textContent = "No speed gates found for current map";
        btn.style.background = "#f44336";
        calibrating = false;
        setTimeout(() => {
          btn.textContent = "Calibrate";
          btn.style.background = "";
        }, 2e3);
        return;
      }
      btn.textContent = "Click on LEFT speed gate";
      btn.style.background = "#e94560";
      overlay.style.pointerEvents = "auto";
      overlay.style.cursor = "crosshair";
      overlay.style.background = "rgba(0,0,0,0.15)";
      overlay.innerHTML = "";
    } else {
      btn.textContent = "Calibrate";
      btn.style.background = "";
      overlay.style.pointerEvents = "none";
      overlay.style.cursor = "";
      overlay.style.background = "";
    }
  });
  document.getElementById("cfOverlay").addEventListener("click", (e) => {
    if (!calibrating) return;
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const xPct = (e.clientX - rect.left) / rect.width * 100;
    const yPct = (e.clientY - rect.top) / rect.height * 100;
    calibrationClicks.push({ x: xPct, y: yPct });
    const btn = document.getElementById("calibrateBtn");
    const overlay = document.getElementById("cfOverlay");
    function crosshairHTML(x, y) {
      return `<div style="position:absolute;left:${x}%;top:0;width:2px;height:100%;background:rgba(255,255,0,0.7);pointer-events:none;"></div>
            <div style="position:absolute;top:${y}%;left:0;width:100%;height:2px;background:rgba(255,255,0,0.7);pointer-events:none;"></div>
            <div style="position:absolute;left:${x}%;top:${y}%;width:16px;height:16px;border-radius:50%;background:yellow;border:2px solid red;transform:translate(-50%,-50%);pointer-events:none;z-index:999;"></div>`;
    }
    if (calibrationClicks.length === 1) {
      overlay.innerHTML = crosshairHTML(xPct, yPct);
      btn.textContent = "Click on RIGHT speed gate";
    } else if (calibrationClicks.length === 2) {
      const ox1 = calibrationClicks[0].x, oy1 = calibrationClicks[0].y;
      const ox2 = calibrationClicks[1].x, oy2 = calibrationClicks[1].y;
      const gx1 = calibrationGates.left.gx, gy1 = calibrationGates.left.gy;
      const gx2 = calibrationGates.right.gx, gy2 = calibrationGates.right.gy;
      const b_x = (ox2 - ox1) / (gx2 - gx1);
      const a_x = ox1 - b_x * gx1;
      const b_y = b_x * 16 / 9;
      const a_y = oy1 - b_y * (1080 - gy1);
      const gameTransform = {
        a_x: Math.round(a_x * 1e3) / 1e3,
        b_x: Math.round(b_x * 1e5) / 1e5,
        a_y: Math.round(a_y * 1e3) / 1e3,
        b_y: Math.round(b_y * 1e5) / 1e5
      };
      if (chapterData) {
        chapterData.game_transform = gameTransform;
      }
      console.log("game_transform:", JSON.stringify(gameTransform));
      const json = JSON.stringify(gameTransform);
      navigator.clipboard.writeText(`"game_transform": ${json}`).then(() => {
        btn.textContent = "Calibrated! (copied to clipboard)";
      }).catch(() => {
        btn.textContent = "Calibrated! Check console";
      });
      btn.style.background = "#4caf50";
      calibrating = false;
      calibrationClicks = [];
      calibrationGates = null;
      overlay.style.pointerEvents = "none";
      overlay.style.cursor = "";
      overlay.style.background = "";
      if (player && player.getCurrentTime) {
        updateOverlay(player.getCurrentTime());
      }
    }
  });
  var urlParams = new URLSearchParams(window.location.search);
  var chaptersUrl = urlParams.get("chapters") || "chapters/tournaments/842.json";
  fetch(chaptersUrl).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }).then(loadChaptersFromJSON).catch((err) => console.log(`Failed to load ${chaptersUrl}: ${err.message}`));
})();
//# sourceMappingURL=player.js.map
