/// <reference path="youtube.d.ts" />
// --- Type definitions ---

type MapName = 'Day' | 'Dusk' | 'Night' | 'Twilight';
type Team = 'gold' | 'blue';
type PositionId = string;

interface AffineTransform {
    a_x: number;
    b_x: number;
    a_y: number;
    b_y: number;
}

interface QueenKill {
    time: number;
    victim: number;
}

interface PlayerEvent {
    time: number;
    delta: number;
    type: string;
    positions?: number[];
    id?: number;
    values?: string[];
    ml_score?: number;
}

interface KillEvent {
    killer: number;
    victim: number;
    time: number;
}

interface TimelinePoint {
    t: number;
    p: number;
}

interface ModelTimelinePoint extends TimelinePoint {
    c?: Record<string, number>;
    eg?: number[];
    ee?: [number, number];
    bg?: (number | null)[];
    bc?: [number, number];
    sx?: number;
}

interface MatchInfo {
    blue: string;
    gold: string;
}

interface Chapter {
    title: string;
    start_time: number;
    end_time: number;
    duration: number;
    map: MapName;
    winner: Team;
    win_condition: string;
    game_id: number;
    set_number: number;
    is_set_start: boolean;
    match_info?: MatchInfo;
    queen_kills?: QueenKill[];
    player_events?: PlayerEvent[];
    kill_events?: KillEvent[];
    win_timeline?: TimelinePoint[];
    model_timelines?: Record<string, ModelTimelinePoint[]>;
    users?: Record<string, string | number>;
    gold_on_left?: boolean;
    hivemind_url: string;
    _timelineFields?: Record<string, boolean>;
}

interface UserInfo {
    name: string;
    scene?: string;
}

interface ChapterData {
    video_id: string | null;
    chapters: Chapter[];
    users: Record<string, UserInfo>;
    game_transform?: AffineTransform;
}

interface FlatQueenKill {
    time: number;
    victim: number;
    game_id: number;
}

interface PlayerHighlight {
    time: number;
    delta: number;
    type: string;
    game_id: number;
    set_number: number;
    event_id?: number;
    values?: string[];
    position: number | null;
    ml_score?: number;
    score?: number;
    id?: number;
}

interface HighImpactRange {
    start: number;
    end: number;
    delta: number;
}

interface KDStats {
    kills: number;
    deaths: number;
}

interface MaidenInfo {
    0: string;  // type: 'maiden_speed' | 'maiden_wings'
    1: number;  // x
    2: number;  // y
}

interface MapStructureInfo {
    maiden_info: [string, number, number][];
    left_berries_centroid: [number, number];
    right_berries_centroid: [number, number];
    snail_center: [number, number];
    blue_hive: [number, number];
    gold_hive: [number, number];
    gold_eggs_centroid: [number, number];
}

interface SpeedGates {
    left: { gx: number; gy: number };
    right: { gx: number; gy: number };
}

interface CalibrationClick {
    x: number;
    y: number;
}

interface OverlayEntry {
    key: string;
    delta: number;
    rawDelta: number;
    x: number;
    y: number;
}

// --- End type definitions ---

// Escape HTML special characters to prevent XSS when inserting into innerHTML.
function esc(s: string): string {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

const chapterList = document.getElementById('chapterList')!;
const currentChapterInfo = document.getElementById('currentChapterInfo')!;
const timeDisplay = document.getElementById('timeDisplay')!;
const chapterFilter = document.getElementById('chapterFilter') as HTMLInputElement;
const playPauseBtn = document.getElementById('playPause')!;

let player: YT.Player | null = null;
let chapters: Chapter[] = [];
let queenKills: FlatQueenKill[] = [];  // Flat list of all queen kill timestamps
let currentChapterIndex = -1;
let lastQueenKillIndex = -1;  // Track last navigated queen kill
let timeUpdateInterval: ReturnType<typeof setInterval> | undefined;

// Favorite team perspective toggle: null = auto (from player selection), 'blue', 'gold'
let favoriteTeam: 'blue' | 'gold' | null = null;

// Player highlight state
let selectedPosition: PositionId | null = null;
let playerHighlights: PlayerHighlight[] = [];  // Filtered events for selected position
let playerHighlightCount = 0;
let playerLowlightCount = 0;
let lastHighlightIndex = -1;

// Tournament user data
let users: Record<string, UserInfo> = {};  // user_id -> {name, scene}
let selectedUserId: string | null = null;  // Currently selected user

// Seconds to seek before a highlight event
const HIGHLIGHT_SEEK_BUFFER = 4.5;

// Y-offset (in overlay %) for maiden gate overlay items
const GATE_Y_OFFSET = 4;

// Vertical spacing in game-space pixels between stacked overlay items
const OVERLAY_LINE_HEIGHT = 20;

// Highlight auto-play mode
let highlightModeEnabled = false;
const HIGHLIGHT_PLAY_DURATION = 6.0;  // Seconds to play each highlight before advancing

// Position names and icons for each character
const POSITION_NAMES: Record<string, string> = {
    '1': 'Gold Queen', '2': 'Blue Queen',
    '3': 'Gold Stripes', '4': 'Gold Skull', '5': 'Gold Abs', '6': 'Gold Checkers',
    '7': 'Blue Stripes', '8': 'Blue Skull', '9': 'Blue Abs', '10': 'Blue Checkers'
};

// Emoji icons (for dropdowns where we can't use HTML)
const POSITION_ICONS: Record<string, string> = {
    '1': '👑',   // Gold Queen
    '2': '👑',   // Blue Queen
    '3': '☰',    // Gold Stripes (trigram)
    '4': '💀',   // Gold Skull
    '5': '💪',   // Gold Abs
    '6': '▦',    // Gold Checkers
    '7': '☰',    // Blue Stripes
    '8': '💀',   // Blue Skull
    '9': '💪',   // Blue Abs
    '10': '▦'    // Blue Checkers
};

// SVG icons for each position type (allows colored images)
const GOLD_COLOR = '#ffc107';
const BLUE_COLOR = '#2196f3';
const DARK_BG = '#1a1a1a';

// Temporary: 'pulse' | 'bold' — press X to toggle, remove after picking winner
let currentCellStyle: 'pulse' | 'bold' = 'pulse';

// SVG templates for each character type (using encodeURIComponent for proper encoding)
const SVG_TEMPLATES: Record<string, (color: string) => string> = {
    // Crown for queens
    crown: (color: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><path d="M5 16L3 5l5.5 5L12 4l3.5 6L21 5l-2 11H5zm14 3c0 .6-.4 1-1 1H6c-.6 0-1-.4-1-1v-1h14v1z"/></svg>`)}`,
    // Horizontal stripes
    stripes: (color: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><rect x="3" y="4" width="18" height="3" rx="1"/><rect x="3" y="10" width="18" height="3" rx="1"/><rect x="3" y="16" width="18" height="3" rx="1"/></svg>`)}`,
    // Skull
    skull: (color: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><path d="M12 2C6.5 2 2 6.5 2 12v3.5c0 1.4 1.1 2.5 2.5 2.5H6v-3h2v4h3v-4h2v4h3v-4h2v3h1.5c1.4 0 2.5-1.1 2.5-2.5V12c0-5.5-4.5-10-10-10zm-3 12a2 2 0 110-4 2 2 0 010 4zm6 0a2 2 0 110-4 2 2 0 010 4z"/></svg>`)}`,
    // Abs/muscular figure
    abs: (color: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><ellipse cx="12" cy="5" rx="4" ry="3"/><path d="M8 9h8v13H8V9z"/><rect x="9" y="10" width="2.5" height="3" fill="${DARK_BG}"/><rect x="12.5" y="10" width="2.5" height="3" fill="${DARK_BG}"/><rect x="9" y="14" width="2.5" height="3" fill="${DARK_BG}"/><rect x="12.5" y="14" width="2.5" height="3" fill="${DARK_BG}"/><rect x="9" y="18" width="2.5" height="2.5" fill="${DARK_BG}"/><rect x="12.5" y="18" width="2.5" height="2.5" fill="${DARK_BG}"/></svg>`)}`,
    // Checkerboard pattern
    checkers: (color: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect width="24" height="24" fill="${color}"/><rect x="0" y="0" width="6" height="6" fill="${DARK_BG}"/><rect x="12" y="0" width="6" height="6" fill="${DARK_BG}"/><rect x="6" y="6" width="6" height="6" fill="${DARK_BG}"/><rect x="18" y="6" width="6" height="6" fill="${DARK_BG}"/><rect x="0" y="12" width="6" height="6" fill="${DARK_BG}"/><rect x="12" y="12" width="6" height="6" fill="${DARK_BG}"/><rect x="6" y="18" width="6" height="6" fill="${DARK_BG}"/><rect x="18" y="18" width="6" height="6" fill="${DARK_BG}"/></svg>`)}`
};

// SVG data URLs for each position
const POSITION_SVGS: Record<string, string> = {
    '1': SVG_TEMPLATES.crown(GOLD_COLOR),    // Gold Queen
    '2': SVG_TEMPLATES.crown(BLUE_COLOR),    // Blue Queen
    '3': SVG_TEMPLATES.stripes(GOLD_COLOR),  // Gold Stripes
    '4': SVG_TEMPLATES.skull(GOLD_COLOR),    // Gold Skull
    '5': SVG_TEMPLATES.abs(GOLD_COLOR),      // Gold Abs
    '6': SVG_TEMPLATES.checkers(GOLD_COLOR), // Gold Checkers
    '7': SVG_TEMPLATES.stripes(BLUE_COLOR),  // Blue Stripes
    '8': SVG_TEMPLATES.skull(BLUE_COLOR),    // Blue Skull
    '9': SVG_TEMPLATES.abs(BLUE_COLOR),      // Blue Abs
    '10': SVG_TEMPLATES.checkers(BLUE_COLOR) // Blue Checkers
};

// Create an img element with the position's SVG icon
function getPositionIconImg(pos: string, size: number = 18): string {
    if (!POSITION_SVGS[pos]) return '';
    return `<img src="${POSITION_SVGS[pos]}" width="${size}" height="${size}" style="vertical-align: middle; margin-right: 4px;" alt="${POSITION_NAMES[pos] || 'Position'}">`;
}

// Get display string for a position (icon + name)
function getPositionDisplay(pos: string, includeIcon: boolean = true, useImage: boolean = false): string {
    const name = POSITION_NAMES[pos] || `Position ${pos}`;
    if (includeIcon) {
        if (useImage && POSITION_SVGS[pos]) {
            return getPositionIconImg(pos) + name;
        } else if (POSITION_ICONS[pos]) {
            return `${POSITION_ICONS[pos]} ${name}`;
        }
    }
    return name;
}

// Video ID from chapters data
let videoId: string | null = null;
let chapterData: ChapterData | null = null;  // top-level JSON data (for game_transform etc.)
let youtubeApiReady = false;

// Initialize YouTube player (called when both API and chapters are ready)
function initializePlayer(): void {
    if (!youtubeApiReady || !videoId || player) return;

    player = new YT.Player('player', {
        videoId: videoId,
        playerVars: {
            'autoplay': 0,
            'controls': 1,
            'rel': 0,
            'modestbranding': 1,
        },
        events: {
            'onReady': onPlayerReady,
            'onStateChange': onPlayerStateChange
        }
    });
}

// Called by YouTube API when ready
function onYouTubeIframeAPIReady(): void {
    youtubeApiReady = true;
    initializePlayer();
}
// Expose to window for YouTube API callback.
// If the API loaded before this script, call initializePlayer directly.
window.onYouTubeIframeAPIReady = onYouTubeIframeAPIReady;
if (window.YT && window.YT.Player) {
    onYouTubeIframeAPIReady();
}

function onPlayerReady(event: YT.PlayerEvent): void {
    console.log('YouTube player ready');
    // Start time update interval
    timeUpdateInterval = setInterval(updateCurrentChapter, 500);

    // Handle URL params: ?game=1714369&t=60
    const params = new URLSearchParams(window.location.search);
    const gameParam = params.get('game');
    const tParam = params.get('t');
    if (gameParam) {
        const gid = parseInt(gameParam);
        const idx = chapters.findIndex(ch => ch.game_id === gid);
        if (idx >= 0) {
            const seekTime = tParam ? chapters[idx].start_time + parseFloat(tParam) : chapters[idx].start_time;
            seekTo(seekTime);
            console.log(`URL nav: game ${gid} (chapter ${idx}), seeking to ${seekTime}s`);
        }
    } else if (tParam) {
        seekTo(parseFloat(tParam));
    }
}

function onPlayerStateChange(event: YT.OnStateChangeEvent): void {
    if (event.data === YT.PlayerState.PLAYING) {
        playPauseBtn.textContent = '⏸ Pause';
    } else {
        playPauseBtn.textContent = '▶ Play';
    }
}

// Get current time from YouTube player
function getCurrentTime(): number {
    return player && player.getCurrentTime ? player.getCurrentTime() : 0;
}

// Get video duration
function getDuration(): number {
    return player && player.getDuration ? player.getDuration() : 0;
}

// Seek to time
function seekTo(seconds: number): void {
    if (player && player.seekTo) {
        player.seekTo(seconds, true);
    }
}

// Play/Pause
function togglePlayPause(): void {
    if (!player) return;
    const state = player.getPlayerState();
    if (state === YT.PlayerState.PLAYING) {
        player.pauseVideo();
    } else {
        player.playVideo();
    }
}

// Format time as M:SS or H:MM:SS
function formatTime(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
        return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// Find chapter at current time
function findChapterAtTime(time: number): number {
    for (let i = chapters.length - 1; i >= 0; i--) {
        if (time >= chapters[i].start_time) {
            return i;
        }
    }
    return -1;
}

// Update current chapter display
function updateCurrentChapter(): void {
    const currentTime = getCurrentTime();
    const newIndex = findChapterAtTime(currentTime);

    if (newIndex !== currentChapterIndex) {
        currentChapterIndex = newIndex;

        // If user is selected, update selectedPosition for this chapter
        if (selectedUserId && currentChapterIndex >= 0) {
            const pos = getUserPositionInChapter(selectedUserId, chapters[currentChapterIndex]);
            selectedPosition = pos ? String(pos) : null;
        }

        // Update sidebar highlighting
        document.querySelectorAll('.chapter-item').forEach((el, i) => {
            el.classList.toggle('active', i === currentChapterIndex);
        });

        // Scroll active chapter into view
        const activeEl = document.querySelector('.chapter-item.active');
        if (activeEl) {
            activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    // Update info panel
    if (currentChapterIndex >= 0) {
        const ch = chapters[currentChapterIndex];
        currentChapterInfo.innerHTML = `<h3>${esc(ch.title)}</h3>
            <span class="${esc(ch.winner)}">${esc(ch.winner)}</span> wins by ${esc(ch.win_condition)}
            &nbsp;|&nbsp; ${formatTime(ch.duration)}
            &nbsp;|&nbsp; <a href="${esc(ch.hivemind_url)}" target="_blank" style="color: #e94560;">HiveMind</a>`;
    }

    // Update time display
    timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(getDuration())}`;

    // Check for highlight auto-advance
    checkHighlightAutoAdvance();

    // Update counterfactual bars
    updateContributionBars(currentTime);

    // Update egg grid
    updateEggGrid(currentTime);

    // Update berry grid
    updateBerryGrid(currentTime);

    // Update map overlay
    updateOverlay(currentTime);
}

// Counterfactual event display labels
const CF_LABELS: Record<string, [string, string | null]> = {
    'bqk': ['Queen Kill', 'blue'],
    'gqk': ['Queen Kill', 'gold'],
    'bb': ['Berry', 'blue'],
    'gb': ['Berry', 'gold'],
    'bswd': ['Speed Warrior Dies', 'blue'],
    'bvwd': ['Warrior Dies', 'blue'],
    'gswd': ['Speed Warrior Dies', 'gold'],
    'gvwd': ['Warrior Dies', 'gold'],
    'bsdw': ['Speed Gets Wings', 'blue'],
    'bdw': ['Gets Wings', 'blue'],
    'bws': ['Gets Speed', 'blue'],
    'gsdw': ['Speed Gets Wings', 'gold'],
    'gdw': ['Gets Wings', 'gold'],
    'gws': ['Gets Speed', 'gold'],
    'sb': ['Snail \u2192 Blue', null],
    'sg': ['Snail \u2192 Gold', null],
};
// Add per-maiden labels dynamically
for (let i = 0; i < 5; i++) {
    CF_LABELS[`mb${i}`] = [`Gate ${i} \u2192 Blue`, 'blue'];
    CF_LABELS[`mg${i}`] = [`Gate ${i} \u2192 Gold`, 'gold'];
}

// Map structure data for overlay positioning (from map_structure_info.json)
// Coordinates are in game space (1920x1080), converted to percentages for overlay
const MAP_STRUCTURE: Record<string, MapStructureInfo> = {
    'Day': {
        maiden_info: [
            ['maiden_speed', 410, 860],
            ['maiden_speed', 1510, 860],
            ['maiden_wings', 560, 260],
            ['maiden_wings', 960, 500],
            ['maiden_wings', 1360, 260]
        ],
        left_berries_centroid: [830, 937],
        right_berries_centroid: [1090, 937],
        snail_center: [960, 1010],
        blue_hive: [1860, 980],
        gold_hive: [60, 980],
        gold_eggs_centroid: [850, 899]
    },
    'Dusk': {
        maiden_info: [
            ['maiden_speed', 340, 140],
            ['maiden_speed', 1580, 140],
            ['maiden_wings', 310, 620],
            ['maiden_wings', 960, 140],
            ['maiden_wings', 1610, 620]
        ],
        left_berries_centroid: [800, 685],
        right_berries_centroid: [1120, 685],
        snail_center: [960, 870],
        blue_hive: [1860, 980],
        gold_hive: [60, 980],
        gold_eggs_centroid: [746, 532]
    },
    'Night': {
        maiden_info: [
            ['maiden_speed', 170, 740],
            ['maiden_speed', 1750, 740],
            ['maiden_wings', 700, 260],
            ['maiden_wings', 960, 700],
            ['maiden_wings', 1220, 260]
        ],
        left_berries_centroid: [170, 96],
        right_berries_centroid: [1750, 96],
        snail_center: [960, 970],
        blue_hive: [1860, 980],
        gold_hive: [60, 980],
        gold_eggs_centroid: [97, 55]
    },
    'Twilight': {
        maiden_info: [
            ['maiden_speed', 410, 860],
            ['maiden_speed', 1510, 860],
            ['maiden_wings', 550, 260],
            ['maiden_wings', 960, 820],
            ['maiden_wings', 1370, 260]
        ],
        left_berries_centroid: [158, 322],
        right_berries_centroid: [1762, 322],
        snail_center: [960, 1010],
        blue_hive: [1860, 980],
        gold_hive: [60, 980],
        gold_eggs_centroid: [164, 52]
    }
};

// Get overlay position for a counterfactual key
// snailX: raw pixel position of snail (from timeline point 'sx'), or null
// cfDict: counterfactual dict (point.c) for gate ownership inference, or null
// Returns: array of [x%, y%] positions, or null if no position available
function getOverlayPosition(key: string, mapInfo: MapStructureInfo, goldOnLeft: boolean | undefined, transform: AffineTransform | undefined, snailX: number | undefined, cfDict: Record<string, number> | undefined): [number, number][] | null {
    if (!mapInfo) return null;

    // Helper: convert game coords (1920x1080) to overlay percentages,
    // using affine transform from landmark calibration if available.
    // Game uses y-up (cartesian) coordinates; flip to screen y-down.
    function toPercent(x: number, y: number, flipX: boolean): [number, number] {
        const px = flipX ? (1920 - x) : x;
        const py = 1080 - y;
        if (transform) {
            return [
                transform.a_x + transform.b_x * px,
                transform.a_y + transform.b_y * py
            ];
        }
        return [px / 1920 * 100, py / 1080 * 100];
    }

    // gold_on_left=True means gold hive is on left side (default orientation)
    // gold_on_left=False means positions are mirrored
    const needsFlip = !goldOnLeft;

    // Per-maiden gates — use exact coordinates from map_structure_info.json
    const maidenMatch = key.match(/^m[bg](\d)$/);
    if (maidenMatch) {
        const idx = parseInt(maidenMatch[1]);
        if (idx < mapInfo.maiden_info.length) {
            const [, mx, my] = mapInfo.maiden_info[idx];
            return [toPercent(mx, my, needsFlip)];
        }
        return null;
    }

    // Berry deposits — canonical positions (gold_on_left=True frame), x-flip handles mirroring
    if (key === 'bb') {
        const c = mapInfo.right_berries_centroid;  // blue's canonical side
        return [toPercent(c[0], c[1], needsFlip)];
    }
    if (key === 'gb') {
        const c = mapInfo.left_berries_centroid;  // gold's canonical side
        return [toPercent(c[0], c[1], needsFlip)];
    }

    // Snail — follows actual snail position from timeline data
    // snail_center y is in screen-space (y-down); convert to y-up for toPercent
    if (key === 'sb' || key === 'sg') {
        const sx = (snailX != null) ? snailX : mapInfo.snail_center[0];
        const sy = 1080 - mapInfo.snail_center[1];
        // Offset sb/sg vertically so they don't overlap
        const yOff = key === 'sb' ? OVERLAY_LINE_HEIGHT : -OVERLAY_LINE_HEIGHT;
        return [toPercent(sx, sy + yOff, false)];  // snailX is already in absolute coords
    }

    // Queen kills — position near team's egg display in top HUD
    // Per-map calibrated positions; blue side derived by symmetry about x=960
    if (key === 'bqk' || key === 'gqk') {
        const gc = mapInfo.gold_eggs_centroid || [850, 899];
        const blueX = 960 + (960 - gc[0]);
        const ey = gc[1] - 45;  // shift down 45px below eggs
        if (key === 'gqk') {
            return [toPercent(goldOnLeft ? gc[0] : blueX, ey, false)];
        }
        return [toPercent(goldOnLeft ? blueX : gc[0], ey, false)];
    }

    // Warrior deaths — near team's hive, offset vertically between vanilla/speed
    if (key === 'bvwd' || key === 'bswd') {
        const h = mapInfo.blue_hive;
        const yOff = key === 'bswd' ? OVERLAY_LINE_HEIGHT : 0;
        return [toPercent(h[0], h[1] + yOff, needsFlip)];
    }
    if (key === 'gvwd' || key === 'gswd') {
        const h = mapInfo.gold_hive;
        const yOff = key === 'gswd' ? OVERLAY_LINE_HEIGHT : 0;
        return [toPercent(h[0], h[1] + yOff, needsFlip)];
    }

    // Wing upgrades — at each team-controlled wings maiden
    if (key === 'bdw' || key === 'bsdw' || key === 'gdw' || key === 'gsdw') {
        const isBlue = key.startsWith('b');
        const yOff = isBlue ? -OVERLAY_LINE_HEIGHT : OVERLAY_LINE_HEIGHT;
        const positions: [number, number][] = [];
        mapInfo.maiden_info.forEach(([type, mx, my], idx) => {
            if (type !== 'maiden_wings') return;
            const flipKey = isBlue ? `mb${idx}` : `mg${idx}`;
            const isControlled = cfDict && !(flipKey in cfDict);
            if (isControlled) {
                positions.push(toPercent(mx, my + yOff, needsFlip));
            }
        });
        return positions.length > 0 ? positions : null;
    }

    // Speed upgrades — at each team-controlled speed maiden
    if (key === 'bws' || key === 'gws') {
        const isBlue = key === 'bws';
        const positions: [number, number][] = [];
        mapInfo.maiden_info.forEach(([type, mx, my], idx) => {
            if (type !== 'maiden_speed') return;
            // Gate is team-controlled if flipping to that team is NOT in cfDict
            // (i.e., flipping would be a no-op because it's already that team's)
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

// Update the map overlay with counterfactual bars at map positions
function updateOverlay(currentTime: number): void {
    const overlay = document.getElementById('cfOverlay');
    if (!overlay) return;

    if (currentChapterIndex < 0) {
        overlay.innerHTML = '';
        return;
    }

    const ch = chapters[currentChapterIndex];
    if (!chapterData || !ch.model_timelines || !ch.map || ch.gold_on_left === undefined) {
        overlay.innerHTML = '';
        return;
    }

    const mapInfo = MAP_STRUCTURE[ch.map];
    if (!mapInfo) {
        overlay.innerHTML = '';
        return;
    }

    const point = findTimelinePoint(ch, currentTime, 'c');
    if (!point || !point.c) {
        overlay.innerHTML = '';
        return;
    }

    const flipForGold = shouldFlipForGold();

    // Collect positioned counterfactual entries
    const positionedEntries = [];
    for (const [key, delta] of Object.entries(point.c)) {
        const displayDelta = flipForGold ? -delta : delta;
        if (Math.abs(displayDelta) < 0.005) continue;
        const positions = getOverlayPosition(key, mapInfo, ch.gold_on_left, chapterData.game_transform, point.sx, point.c);
        if (!positions) continue;
        const isGate = /^m[bg]\d$/.test(key);
        for (const pos of positions) {
            positionedEntries.push({ key, delta: displayDelta, rawDelta: delta, x: pos[0], y: pos[1] + (isGate ? GATE_Y_OFFSET : 0) });
        }
    }

    if (positionedEntries.length === 0) {
        overlay.innerHTML = '';
        return;
    }

    const maxDelta = Math.max(0.10, ...positionedEntries.map(e => Math.abs(e.delta)));

    // Blue = good for blue team, orange = good for gold team
    const BLUE_CF = 'rgba(59,130,246,0.9)';
    const ORANGE_CF = 'rgba(249,115,22,0.9)';

    let html = '';
    for (const e of positionedEntries) {
        const fillPct = (Math.abs(e.delta) / maxDelta) * 50; // max 50% of bar width
        const isBlueGood = e.rawDelta > 0;  // positive raw delta = good for blue
        const color = isBlueGood ? BLUE_CF : ORANGE_CF;
        const goesRight = barGoesRight(e.rawDelta, e.delta, ch.gold_on_left);
        const barStyle = goesRight
            ? `left:50%;width:${fillPct}%;background:${color};`
            : `right:50%;width:${fillPct}%;background:${color};`;
        const pctText = `${e.key} ${(Math.abs(e.delta) * 100).toFixed(0)}%`;
        const labelColor = isBlueGood ? '#93c5fd' : '#fdba74';

        html += `<div class="cf-overlay-item" style="left:${e.x}%;top:${e.y}%;">
            <span class="cf-overlay-label" style="color:${labelColor}">${pctText}</span>
            <div class="cf-overlay-bar">
                <div class="cf-overlay-bar-fill" style="${barStyle}"></div>
            </div>
        </div>`;
    }

    overlay.innerHTML = html;
}

// Find the closest timeline point to a given time using binary search
function findClosestPoint(timeline: ModelTimelinePoint[], time: number): ModelTimelinePoint | null {
    if (!timeline || timeline.length === 0) return null;
    let lo = 0, hi = timeline.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (timeline[mid].t < time) lo = mid + 1;
        else hi = mid;
    }
    // Check neighbors for closest
    if (lo > 0 && Math.abs(timeline[lo - 1].t - time) < Math.abs(timeline[lo].t - time)) {
        lo--;
    }
    return timeline[lo];
}

// Find the closest point in the first model timeline that has the given field.
// Caches which timelines have which fields to avoid repeated .some() scans.
function findTimelinePoint(ch: Chapter, currentTime: number, field: string): ModelTimelinePoint | null {
    if (!ch.model_timelines) return null;
    if (!ch._timelineFields) ch._timelineFields = {};
    for (const name of Object.keys(ch.model_timelines)) {
        const timeline = ch.model_timelines[name];
        if (!timeline || timeline.length === 0) continue;
        const cacheKey = name + ':' + field;
        if (!(cacheKey in ch._timelineFields)) {
            ch._timelineFields[cacheKey] = timeline.some(pt => (pt as unknown as Record<string, unknown>)[field]);
        }
        if (!ch._timelineFields[cacheKey]) continue;
        return findClosestPoint(timeline, currentTime);
    }
    return null;
}

// Update the contribution bars display
function updateContributionBars(currentTime: number): void {
    const container = document.getElementById('contributionBars')!;
    const content = document.getElementById('cfBarsContent')!;

    if (currentChapterIndex < 0) {
        container.style.display = 'none';
        return;
    }

    const ch = chapters[currentChapterIndex];
    if (!ch.model_timelines) {
        container.style.display = 'none';
        return;
    }

    const point = findTimelinePoint(ch, currentTime, 'c');
    if (!point || !point.c) {
        container.style.display = 'none';
        return;
    }

    container.style.display = '';

    const flipForGold = shouldFlipForGold();

    // Build sorted entries
    const entries = [];
    for (const [key, delta] of Object.entries(point.c)) {
        const displayDelta = flipForGold ? -delta : delta;
        if (Math.abs(displayDelta) < 0.005) continue;  // hide < 0.5%
        const labelInfo = CF_LABELS[key] || [key, null];
        entries.push({ key, label: `${key} ${labelInfo[0]}`, team: labelInfo[1], delta: displayDelta, rawDelta: delta });
    }
    entries.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

    // Max bar scale: use the largest absolute delta, minimum 0.10
    const maxDelta = Math.max(0.10, ...entries.map(e => Math.abs(e.delta)));

    let html = '';
    for (const e of entries) {
        const pct = (Math.abs(e.delta) / maxDelta) * 50;  // max 50% of track width
        const barClass = e.rawDelta > 0 ? 'positive' : 'negative';
        const goesRight = barGoesRight(e.rawDelta, e.delta, ch.gold_on_left);
        const barStyle = goesRight
            ? `left: 50%; width: ${pct}%;`
            : `left: ${50 - pct}%; width: ${pct}%;`;

        const teamColor = e.team === 'blue' ? '#5ba3ec'
            : e.team === 'gold' ? '#ffd700' : '#aaa';
        const valueStr = (Math.abs(e.delta) * 100).toFixed(1) + '%';

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

// Color interpolation: blue rgba(59,130,246) for high prob, orange rgba(249,115,22) for low.
function probToColor(prob: number): string {
    const r = Math.round(249 + (59 - 249) * prob);
    const g = Math.round(115 + (130 - 115) * prob);
    const b = Math.round(22 + (246 - 22) * prob);
    return `rgba(${r},${g},${b},0.9)`;
}

// Return per-edge contour border CSS for edges where the 50% boundary crosses.
// rotate(45deg) maps CSS borders to diamond edges:
//   CSS border-top    → upper-right diamond edge (neighbor depends on mirror)
//   CSS border-right  → lower-right diamond edge
//   CSS border-bottom → lower-left diamond edge
//   CSS border-left   → upper-left diamond edge
function contourBorderCSS(
    probs: (number | null)[][], row: number, col: number, n: number, needsMirror: boolean
): string {
    const p = probs[row][col];
    if (p === null) return '';
    const side = p >= 0.5;

    // Grid neighbors: up(row-1), down(row+1), left(col-1), right(col+1)
    function crosses(r: number, c: number): boolean {
        if (r < 0 || r >= n || c < 0 || c >= n) return false;
        const np = probs[r][c];
        if (np === null) return false;
        return (np >= 0.5) !== side;
    }

    // In the diamond layout, dx = col - row, dy = col + row.
    // Without mirror:
    //   row-1 neighbor is up-right in diamond space → CSS border-top
    //   col+1 neighbor is down-right               → CSS border-right
    //   row+1 neighbor is down-left                 → CSS border-bottom
    //   col-1 neighbor is up-left                   → CSS border-left
    // With mirror (dx negated):
    //   row-1 neighbor is up-left  → CSS border-left
    //   col+1 neighbor is down-left → CSS border-bottom
    //   row+1 neighbor is down-right → CSS border-right
    //   col-1 neighbor is up-right  → CSS border-top
    const borderStyle = '2px solid rgba(255,255,255,0.85)';
    const parts: string[] = [];

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

    return parts.join('');
}

// Render a diamond-shaped grid with rotated cells.
// probs: n*n array of probabilities (null entries are skipped).
// currentRow/currentCol: position of the current-state cell (-1 if none).
// needsMirror: if true, negate dx to swap left/right.
// cellSize: size of each diamond cell in pixels.
// fontSize: font size for the probability text.
// leftLabel/rightLabel: labels for the left and right tips of the diamond.
function renderDiamondGrid(
    probs: (number | null)[][], n: number,
    currentRow: number, currentCol: number,
    needsMirror: boolean, cellSize: number, fontSize: number,
    leftLabel: string, rightLabel: string
): string {
    const step = cellSize * Math.SQRT2 / 2;
    const halfCell = cellSize / 2;
    // dx = col - row (horizontal), dy = col + row (vertical)
    // step = center-to-center distance for edge-sharing rotated squares
    const diagSpan = 2 * n - 1;
    const containerWidth = diagSpan * step + cellSize + 20;
    const containerHeight = diagSpan * step + cellSize;
    const cx = containerWidth / 2;
    const topPad = halfCell; // offset from top

    let html = `<div class="diamond-grid-container" style="width:${containerWidth}px;height:${containerHeight}px;">`;

    for (let row = 0; row < n; row++) {
        for (let col = 0; col < n; col++) {
            const prob = probs[row][col];
            if (prob === null || prob === undefined) continue;

            let dx = col - row;
            if (needsMirror) dx = -dx;
            const dy = col + row;

            const x = cx + dx * step - halfCell;
            const y = topPad + dy * step - halfCell;

            const pct = Math.round(prob * 100);
            const bgColor = probToColor(prob);
            const isCurrent = (row === currentRow && col === currentCol);
            const currentClass = isCurrent
                ? (currentCellStyle === 'pulse' ? ' egg-current-pulse' : ' egg-current-bold')
                : '';
            const contour = contourBorderCSS(probs, row, col, n, needsMirror);

            html += `<div class="diamond-cell${currentClass}" style="left:${x}px;top:${y}px;width:${cellSize + 1}px;height:${cellSize + 1}px;background:${bgColor};${contour}"><span style="font-size:${fontSize}px;">${pct}</span></div>`;
        }
    }

    // Axis labels at left and right tips of the diamond
    const leftX = cx - (n - 0.5) * step;
    const rightX = cx + (n - 0.5) * step;
    const midY = topPad + (n - 1) * step;
    html += `<span class="diamond-axis-label" style="right:${containerWidth - leftX + 4}px;top:${midY - 7}px;">${leftLabel}</span>`;
    html += `<span class="diamond-axis-label" style="left:${rightX + 4}px;top:${midY - 7}px;">${rightLabel}</span>`;

    html += '</div>';
    return html;
}

// Update the 3x3 egg counterfactual grid
function updateEggGrid(currentTime: number): void {
    const container = document.getElementById('eggGrid')!;
    const content = document.getElementById('eggGridContent')!;

    if (currentChapterIndex < 0) {
        container.style.display = 'none';
        return;
    }

    const ch = chapters[currentChapterIndex];
    if (!ch.model_timelines) {
        container.style.display = 'none';
        return;
    }

    const point = findTimelinePoint(ch, currentTime, 'eg');
    if (!point || !point.eg) {
        container.style.display = 'none';
        return;
    }

    container.style.display = '';

    const flipForGold = shouldFlipForGold();
    const needsMirror = !!ch.gold_on_left;

    const eg = point.eg;  // 9-element array indexed as [blue_eggs * 3 + gold_eggs]
    const ee = point.ee;  // [current_blue_eggs, current_gold_eggs]

    // Build 3x3 probability grid
    // row = blue eggs, col = gold eggs (fixed orientation)
    const n = 3;
    const eggProbs: (number | null)[][] = [];
    let currentRow = -1, currentCol = -1;
    for (let row = 0; row < n; row++) {
        eggProbs[row] = [];
        for (let col = 0; col < n; col++) {
            const blueEggs = row;
            const goldEggs = col;
            const idx = blueEggs * n + goldEggs;
            let prob = eg[idx];
            if (flipForGold) prob = 1 - prob;
            eggProbs[row][col] = prob;
            if (ee && blueEggs === ee[0] && goldEggs === ee[1]) {
                currentRow = row;
                currentCol = col;
            }
        }
    }

    // Axis labels based on spatial layout (which team is on which side of the video)
    const leftTeam = ch.gold_on_left ? 'Gold' : 'Blue';
    const rightTeam = ch.gold_on_left ? 'Blue' : 'Gold';
    const leftLabel = `${leftTeam} +eggs`;
    const rightLabel = `${rightTeam} +eggs`;

    content.innerHTML = renderDiamondGrid(eggProbs, n, currentRow, currentCol, needsMirror, 60, 14, leftLabel, rightLabel);
}

const BERRY_DELTAS = [0, 1, 2, 3, 4];
const MAX_FOOD = 12;

// Update the 5x5 berry counterfactual grid
function updateBerryGrid(currentTime: number): void {
    const container = document.getElementById('berryGrid')!;
    const content = document.getElementById('berryGridContent')!;

    if (currentChapterIndex < 0) {
        container.style.display = 'none';
        return;
    }

    const ch = chapters[currentChapterIndex];
    if (!ch.model_timelines) {
        container.style.display = 'none';
        return;
    }

    const point = findTimelinePoint(ch, currentTime, 'bg');
    if (!point || !point.bg) {
        container.style.display = 'none';
        return;
    }

    container.style.display = '';

    const flipForGold = shouldFlipForGold();
    const needsMirror = !!ch.gold_on_left;

    const bg = point.bg;  // 25-element array indexed as [blue_delta * 5 + gold_delta]
    const n = BERRY_DELTAS.length;

    // Build n*n probability grid, skipping null entries (game-over states)
    // row = blue delta, col = gold delta (fixed orientation)
    const berryProbs: (number | null)[][] = [];
    for (let row = 0; row < n; row++) {
        berryProbs[row] = [];
        for (let col = 0; col < n; col++) {
            const blueDelta = row;
            const goldDelta = col;
            const idx = blueDelta * n + goldDelta;
            const raw = bg[idx];
            if (raw === null || raw === undefined) {
                berryProbs[row][col] = null;
            } else {
                berryProbs[row][col] = flipForGold ? 1 - raw : raw;
            }
        }
    }

    // Delta (0,0) is always the current game state
    const currentRow = 0, currentCol = 0;

    // Axis labels based on spatial layout (which team is on which side of the video)
    const leftTeam = ch.gold_on_left ? 'Gold' : 'Blue';
    const rightTeam = ch.gold_on_left ? 'Blue' : 'Gold';
    const leftLabel = `${leftTeam} +berries`;
    const rightLabel = `${rightTeam} +berries`;

    content.innerHTML = renderDiamondGrid(berryProbs, n, currentRow, currentCol, needsMirror, 40, 11, leftLabel, rightLabel);
}

// Calculate net win probability change for a player in a chapter
function calculateNetWinProb(ch: Chapter, positionId: string): number | null {
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

// Find high-impact time ranges for a player in a chapter
function findHighImpactRanges(ch: Chapter, positionId: string): HighImpactRange[] {
    if (!positionId || !ch.player_events) return [];
    const pos = parseInt(positionId);

    // Get all events for this player sorted by time
    const playerEvents = ch.player_events
        .filter(evt => evt.positions && evt.positions.includes(pos) && Math.abs(evt.delta) >= 0.05)
        .sort((a, b) => a.time - b.time);

    if (playerEvents.length === 0) return [];

    // Cluster nearby events (within 5 seconds)
    const ranges: HighImpactRange[] = [];
    let rangeStart: number | null = null;
    let rangeEnd: number | null = null;
    let rangeDelta = 0;

    for (const evt of playerEvents) {
        if (rangeStart === null) {
            rangeStart = evt.time;
            rangeEnd = evt.time;
            rangeDelta = evt.delta;
        } else if (evt.time - rangeEnd! <= 5) {
            rangeEnd = evt.time;
            rangeDelta += evt.delta;
        } else {
            if (Math.abs(rangeDelta) >= 0.10) {
                ranges.push({ start: rangeStart, end: rangeEnd!, delta: rangeDelta });
            }
            rangeStart = evt.time;
            rangeEnd = evt.time;
            rangeDelta = evt.delta;
        }
    }
    // Don't forget the last range
    if (rangeStart !== null && Math.abs(rangeDelta) >= 0.10) {
        ranges.push({ start: rangeStart, end: rangeEnd!, delta: rangeDelta });
    }

    return ranges;
}

// Calculate K/D stats for a chapter and selected position
function calculateKD(ch: Chapter, positionId: string): KDStats | null {
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

// Check if a position is on the gold team
function isGoldTeam(positionId: string): boolean {
    const pos = parseInt(positionId);
    // Gold: odd positions (1, 3, 5, 7, 9)
    // Blue: even positions (2, 4, 6, 8, 10)
    return pos % 2 === 1;
}

// Get the selected player's position in a chapter (or current chapter).
function getChapterPosition(ch?: Chapter | null): string | null {
    if (!ch) ch = currentChapterIndex >= 0 ? chapters[currentChapterIndex] : null;
    if (!ch) return null;
    let pos = selectedPosition;
    if (selectedUserId) {
        const p = getUserPositionInChapter(selectedUserId, ch);
        pos = p ? String(p) : null;
    }
    return pos;
}

// Determine if the current perspective should be flipped to gold's viewpoint.
function shouldFlipForGold(ch?: Chapter): boolean {
    if (favoriteTeam === 'blue') return false;
    if (favoriteTeam === 'gold') return true;
    // Auto: use player's team
    const pos = getChapterPosition(ch);
    return !!pos && isGoldTeam(pos);
}

// Flip a delta from blue's perspective based on a position.
// Returns the delta as seen from the viewer's team.
function perspectiveDelta(delta: number, position: number | string | null): number {
    return (position && isGoldTeam(String(position))) ? -delta : delta;
}

// Determine if a bar should extend rightward based on spatial layout.
function barGoesRight(rawDelta: number, displayDelta: number, goldOnLeft: boolean | undefined): boolean {
    return goldOnLeft !== undefined
        ? (rawDelta > 0) === !!goldOnLeft
        : displayDelta > 0;
}

// Build SVG path string from a timeline array
function buildTimelinePath(timeline: TimelinePoint[], startTime: number, duration: number, width: number, height: number, padding: number, flipForGold: boolean): string {
    let pathD = '';
    for (let i = 0; i < timeline.length; i++) {
        const pt = timeline[i];
        const x = padding + ((pt.t - startTime) / duration) * (width - 2 * padding);
        const prob = flipForGold ? (1 - pt.p) : pt.p;
        const y = height - padding - (prob * (height - 2 * padding));
        if (i === 0) {
            pathD += `M ${x} ${y}`;
        } else {
            pathD += ` L ${x} ${y}`;
        }
    }
    return pathD;
}

// Model comparison colors
const MODEL_COLORS = ['#e94560', '#5ba3ec', '#50c878', '#f5a623'];

// Render win probability plot for a chapter
function renderWinProbPlot(ch: Chapter, index: number): string {
    const hasTimeline = ch.win_timeline && ch.win_timeline.length >= 2;
    const hasModels = ch.model_timelines && Object.keys(ch.model_timelines).length > 0;
    if (!hasTimeline && !hasModels) return '';

    const width = 280;
    const height = 36;
    const padding = 2;

    const startTime = ch.start_time;
    const endTime = ch.end_time;
    const duration = endTime - startTime;

    // Get position for this chapter (may vary if user is selected)
    const chapterPosition = getChapterPosition(ch);
    const flipForGold = shouldFlipForGold(ch);

    // Find high-impact ranges for selected player
    const highImpactRanges = chapterPosition ? findHighImpactRanges(ch, chapterPosition) : [];

    // Create highlight rectangles for high-impact ranges
    const highlightsHtml = highImpactRanges.map(range => {
        const x1 = padding + ((range.start - startTime) / duration) * (width - 2 * padding);
        const x2 = padding + ((range.end - startTime) / duration) * (width - 2 * padding);
        const rangeWidth = Math.max(x2 - x1, 6);
        const isGoodForPlayer = flipForGold ? range.delta < 0 : range.delta > 0;
        const color = isGoodForPlayer ? 'rgba(76, 175, 80, 0.4)' : 'rgba(244, 67, 54, 0.4)';
        return `<rect x="${x1 - 2}" y="0" width="${rangeWidth + 4}" height="${height}" fill="${color}"/>`;
    }).join('');

    let pathsHtml = '';
    let legendHtml = '';

    if (hasModels) {
        const modelNames = Object.keys(ch.model_timelines!);

        // HiveMind baseline as gray dashed line (if available)
        if (hasTimeline) {
            const basePathD = buildTimelinePath(ch.win_timeline!, startTime, duration, width, height, padding, flipForGold);
            pathsHtml += `<path d="${basePathD}" fill="none" stroke="#888" stroke-width="1" stroke-dasharray="3,2" opacity="0.6"/>`;
        }

        // Model curves
        modelNames.forEach((name, idx) => {
            const timeline = ch.model_timelines![name];
            if (!timeline || timeline.length < 2) return;
            const color = MODEL_COLORS[idx % MODEL_COLORS.length];
            const pathD = buildTimelinePath(timeline, startTime, duration, width, height, padding, flipForGold);
            pathsHtml += `<path d="${pathD}" fill="none" stroke="${color}" stroke-width="1.5"/>`;
        });

        // Legend
        const legendItems = modelNames.map((name, idx) => {
            const color = MODEL_COLORS[idx % MODEL_COLORS.length];
            return `<span style="color:${color}; margin-right:8px; font-size:10px;">\u25CF ${name}</span>`;
        });
        if (hasTimeline) {
            legendItems.push('<span style="color:#888; font-size:10px;">\u2504 HiveMind</span>');
        }
        legendHtml = `<div style="display:flex; flex-wrap:wrap; gap:2px; margin-top:2px;">${legendItems.join('')}</div>`;
    } else {
        // Original single-curve rendering
        const pathD = buildTimelinePath(ch.win_timeline!, startTime, duration, width, height, padding, flipForGold);
        pathsHtml = `<path d="${pathD}" fill="none" stroke="#888" stroke-width="1.5"/>`;
    }

    return `
        <div class="win-prob-plot" data-chapter="${index}">
            <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
                ${highlightsHtml}
                <line x1="${padding}" y1="${height/2}" x2="${width-padding}" y2="${height/2}"
                      stroke="#333" stroke-width="1" stroke-dasharray="2,2"/>
                ${pathsHtml}
            </svg>
            ${legendHtml}
        </div>
    `;
}

// Render chapter list
function renderChapters(filter: string = ''): void {
    const filterLower = filter.toLowerCase();

    chapterList.innerHTML = chapters
        .map((ch, i) => {
            // Calculate player's position in this chapter first (needed for filtering)
            const chapterPosition = getChapterPosition(ch);

            // Filter by selected player - skip games they're not in
            if (selectedUserId && !chapterPosition) {
                return '';
            }

            // Filter by text search
            if (filter) {
                const searchText = `${ch.map} ${ch.winner} ${ch.win_condition} ${ch.game_id}`.toLowerCase();
                if (!searchText.includes(filterLower)) {
                    return '';
                }
            }

            const winnerClass = ch.winner === 'gold' ? 'gold-win' : 'blue-win';
            const activeClass = i === currentChapterIndex ? 'active' : '';
            const setClass = ch.is_set_start ? 'set-start' : 'in-set';
            const setLabel = ch.is_set_start && ch.match_info ?
                `<div class="set-label"><span class="blue">${esc(ch.match_info.blue)}</span> vs <span class="gold">${esc(ch.match_info.gold)}</span></div>` : '';

            const plotHtml = renderWinProbPlot(ch, i);

            // Calculate K/D and net win prob for selected player
            let statsHtml = '';
            if (chapterPosition) {
                const kd = calculateKD(ch, chapterPosition);
                const netProb = calculateNetWinProb(ch, chapterPosition);
                if (kd) {
                    // Flip net prob for gold team (delta is from blue's perspective)
                    const playerNetProb = isGoldTeam(chapterPosition) ? -netProb! : netProb;
                    const netProbStr = playerNetProb !== null ?
                        `<span class="${playerNetProb >= 0 ? 'good-prob' : 'bad-prob'}">${playerNetProb >= 0 ? '+' : ''}${(playerNetProb * 100).toFixed(0)}%</span>` : '';
                    statsHtml = `<span class="kd-stats">${kd.kills}/${kd.deaths}</span> ${netProbStr}`;
                }
            }

            return `
                <div class="chapter-item ${winnerClass} ${activeClass} ${setClass}" data-index="${i}">
                    ${setLabel}
                    <div class="chapter-title">${esc(ch.title)}</div>
                    <div class="chapter-meta">
                        <span class="winner ${esc(ch.winner)}">${esc(ch.winner)}</span> ${esc(ch.win_condition)}
                        &nbsp;|&nbsp; ${formatTime(ch.duration)}
                        ${statsHtml ? '&nbsp;|&nbsp;' + statsHtml : ''}
                    </div>
                    ${plotHtml}
                </div>
            `;
        })
        .join('');

    // Add click handlers for chapter items (not on plot)
    document.querySelectorAll<HTMLElement>('.chapter-item').forEach(el => {
        el.addEventListener('click', (e) => {
            // Don't trigger if clicking on plot
            if ((e.target as HTMLElement).closest('.win-prob-plot')) return;
            const index = parseInt(el.dataset.index!);
            jumpToChapter(index);
        });
    });

    // Add click handlers for plots (seek to specific time)
    document.querySelectorAll<HTMLElement>('.win-prob-plot').forEach(plot => {
        plot.addEventListener('click', (e: MouseEvent) => {
            e.stopPropagation();
            const chapterIndex = parseInt(plot.dataset.chapter!);
            const ch = chapters[chapterIndex];
            const rect = plot.getBoundingClientRect();
            const clickX = (e.clientX - rect.left) / rect.width;
            const targetTime = ch.start_time + clickX * (ch.end_time - ch.start_time);
            seekTo(targetTime);
            if (player) player.playVideo();
        });
    });
}

// Jump to chapter
function jumpToChapter(index: number): void {
    if (index >= 0 && index < chapters.length) {
        const targetTime = chapters[index].start_time;
        console.log(`Jumping to chapter ${index}: ${chapters[index].title} at ${targetTime}s`);
        seekTo(targetTime);
        currentChapterIndex = index;
        updateCurrentChapter();
        if (player) player.playVideo();
    }
}

// Navigation
function prevChapter(): void {
    const idx = findChapterAtTime(getCurrentTime());
    if (idx > 0) {
        jumpToChapter(idx - 1);
    } else if (idx === 0) {
        jumpToChapter(0);
    }
}

function nextChapter(): void {
    const idx = findChapterAtTime(getCurrentTime());
    if (idx < chapters.length - 1) {
        jumpToChapter(idx + 1);
    } else if (idx === -1 && chapters.length > 0) {
        jumpToChapter(0);
    }
}

// Set navigation
function nextSet(): void {
    const idx = findChapterAtTime(getCurrentTime());
    for (let i = idx + 1; i < chapters.length; i++) {
        if (chapters[i].is_set_start) {
            jumpToChapter(i);
            return;
        }
    }
}

function prevSet(): void {
    const idx = findChapterAtTime(getCurrentTime());
    // If we're past the start of current set's first game, go to current set start
    const currentSetStart = chapters.findIndex((ch, i) =>
        i <= idx && ch.is_set_start &&
        (i === idx || !chapters.slice(i + 1, idx + 1).some(c => c.is_set_start))
    );

    // Find previous set start
    for (let i = idx - 1; i >= 0; i--) {
        if (chapters[i].is_set_start) {
            // If we're at the start of current set, go to previous set
            if (i === currentSetStart && getCurrentTime() - chapters[i].start_time < 2) {
                continue;
            }
            jumpToChapter(i);
            return;
        }
    }
    // If no previous set, go to first chapter
    if (chapters.length > 0) {
        jumpToChapter(0);
    }
}

// Queen kill (egg) navigation
function findQueenKillIndexAtTime(time: number): number {
    // Find the queen kill at or just before the given time
    for (let i = queenKills.length - 1; i >= 0; i--) {
        if (queenKills[i].time <= time + 2) {
            return i;
        }
    }
    return -1;
}

function nextQueenKill(): void {
    if (queenKills.length === 0) return;

    const currentTime = getCurrentTime();
    let nextIndex;

    // If we're near the last jumped-to kill, go to the next one
    if (lastQueenKillIndex >= 0 && lastQueenKillIndex < queenKills.length - 1) {
        const lastKillTime = queenKills[lastQueenKillIndex].time;
        if (Math.abs(currentTime - (lastKillTime - 1)) < 3) {
            nextIndex = lastQueenKillIndex + 1;
        }
    }

    // Otherwise find based on current time
    if (nextIndex === undefined) {
        nextIndex = findQueenKillIndexAtTime(currentTime) + 1;
    }

    if (nextIndex < queenKills.length) {
        lastQueenKillIndex = nextIndex;
        seekTo(queenKills[nextIndex].time - 1);
        if (player) player.playVideo();
    }
}

function prevQueenKill(): void {
    if (queenKills.length === 0) return;

    const currentTime = getCurrentTime();

    // If we have a last jumped-to kill, check if we should replay it or go back
    if (lastQueenKillIndex >= 0) {
        const lastKillTime = queenKills[lastQueenKillIndex].time;
        const targetTime = lastKillTime - 1;

        // If we're past the kill moment, replay this kill
        if (currentTime > lastKillTime - 0.5) {
            seekTo(targetTime);
            if (player) player.playVideo();
            return;
        }

        // If we're near the start (within 2s), go to previous kill
        if (Math.abs(currentTime - targetTime) < 2 && lastQueenKillIndex > 0) {
            lastQueenKillIndex = lastQueenKillIndex - 1;
            seekTo(queenKills[lastQueenKillIndex].time - 1);
            if (player) player.playVideo();
            return;
        }
    }

    // Otherwise find based on current time
    const idx = findQueenKillIndexAtTime(currentTime);
    if (idx >= 0) {
        lastQueenKillIndex = idx;
        seekTo(queenKills[idx].time - 1);
        if (player) player.playVideo();
    }
}

// Player highlight functions
function updatePlayerHighlights(): void {
    playerHighlights = [];
    lastHighlightIndex = -1;

    // Collect events - either for selected player/position, or all high-impact events
    let allEvents: PlayerHighlight[] = [];
    let anyQueen = false;
    const noSelection = !selectedUserId && !selectedPosition;

    for (const ch of chapters) {
        if (!ch.player_events) continue;

        if (noSelection) {
            // No selection - collect ALL high-impact events (prefer ML-scored)
            for (const evt of ch.player_events) {
                // Include if has ML score OR high delta
                if (evt.ml_score !== undefined || Math.abs(evt.delta) >= 0.10) {
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
            // User or position selected - filter by that
            let pos;
            if (selectedUserId) {
                pos = getUserPositionInChapter(selectedUserId, ch);
                if (!pos) continue;  // User not in this chapter
            } else {
                pos = parseInt(selectedPosition!);
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

    // Queen positions have higher thresholds (bigger swings are normal)
    const baseThreshold = anyQueen ? 0.20 : 0.15;

    // Sort by time
    allEvents.sort((a, b) => a.time - b.time);

    // Calculate impact score - prefer ML score when available
    const windowSize = 5; // seconds
    for (let i = 0; i < allEvents.length; i++) {
        const evt = allEvents[i];

        // Use ML score if available (scale: 1-4, normalize to 0-1 range)
        if (evt.ml_score !== undefined) {
            // ML score 1-4 -> normalized 0-0.75, plus clustering bonus
            evt.score = (evt.ml_score - 1) / 4;
        } else {
            // Fallback: use delta with clustering bonus
            let clusterScore = Math.abs(evt.delta);
            for (let j = 0; j < allEvents.length; j++) {
                if (i !== j && Math.abs(evt.time - allEvents[j].time) < windowSize) {
                    clusterScore += Math.abs(allEvents[j].delta) * 0.3;
                }
            }
            evt.score = clusterScore;
        }
    }

    // Filter by threshold and take top events per set
    // Use lower threshold for ML-scored events (they're pre-filtered)
    const targetPerSet = 4;
    const eventsBySet: Record<number, PlayerHighlight[]> = {};

    for (const evt of allEvents) {
        const threshold = evt.ml_score !== undefined ? 0.05 : baseThreshold;
        if (evt.score! >= threshold) {
            if (!eventsBySet[evt.set_number]) {
                eventsBySet[evt.set_number] = [];
            }
            eventsBySet[evt.set_number].push(evt);
        }
    }

    // Take top N per set by score, with deduplication of nearby events
    const MIN_HIGHLIGHT_GAP = 10;  // Minimum seconds between highlights
    for (const setNum in eventsBySet) {
        eventsBySet[setNum].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
        const selected: PlayerHighlight[] = [];
        for (const evt of eventsBySet[setNum]) {
            // Check if too close to an already-selected event
            const tooClose = selected.some(s => Math.abs(s.time - evt.time) < MIN_HIGHLIGHT_GAP);
            if (!tooClose) {
                selected.push(evt);
                if (selected.length >= targetPerSet) break;
            }
        }
        playerHighlights.push(...selected);
    }

    // Check if there are any positive highlights for this player
    // If not, include their best positive move regardless of threshold
    const hasPositiveHighlight = playerHighlights.some(h => {
        return perspectiveDelta(h.delta, h.position) > 0;
    });

    if (!hasPositiveHighlight && allEvents.length > 0) {
        // Find the best positive move (highest delta from player's perspective)
        let bestPositiveMove = null;
        let bestPositiveDelta = 0;

        for (const evt of allEvents) {
            const displayDelta = perspectiveDelta(evt.delta, evt.position);
            if (displayDelta > bestPositiveDelta) {
                bestPositiveDelta = displayDelta;
                bestPositiveMove = evt;
            }
        }

        if (bestPositiveMove && !playerHighlights.some(h => h.id === bestPositiveMove.id)) {
            playerHighlights.push(bestPositiveMove);
        }
    }

    // Sort final list by time for navigation
    playerHighlights.sort((a, b) => a.time - b.time);

    // Count highlights (good for player) vs lowlights (bad for player)
    playerHighlightCount = 0;
    playerLowlightCount = 0;
    for (const h of playerHighlights) {
        if (perspectiveDelta(h.delta, h.position) >= 0) playerHighlightCount++;
        else playerLowlightCount++;
    }
    document.getElementById('highlightCount')!.innerHTML =
        `<span class="good-prob">${playerHighlightCount}</span> / <span class="bad-prob">${playerLowlightCount}</span>`;

    // Update debug UI
    renderHighlightDebug();
}

// Render the highlight debug panel
function renderHighlightDebug(): void {
    const debugEl = document.getElementById('highlightDebug')!;

    if (!selectedUserId && !selectedPosition) {
        debugEl.innerHTML = '';
        return;
    }

    if (playerHighlights.length === 0) {
        debugEl.innerHTML = '<p class="highlight-debug-empty">No highlights found for this player</p>';
        return;
    }

    // Get player name and icon for header
    let playerName = 'Selected Player';
    let playerIcon = '';
    if (selectedUserId && users[selectedUserId]) {
        playerName = users[selectedUserId].name;
        // Get the position icon for the current chapter
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
        const deltaClass = displayDelta >= 0 ? 'positive' : 'negative';
        const deltaStr = (displayDelta >= 0 ? '+' : '') + (displayDelta * 100).toFixed(0) + '%';
        const scoreStr = h.score ? `(${(h.score * 100).toFixed(0)})` : '';
        const valuesStr = h.values ? h.values.join(', ') : '';
        const eventIdStr = h.event_id ? `#${h.event_id}` : '';

        // Add position icon for this specific event
        const posIcon = h.position ? getPositionIconImg(String(h.position), 14) : '';

        return `
            <div class="highlight-debug-item" data-highlight-index="${idx}">
                <span class="highlight-debug-pos">${posIcon}</span>
                <span class="highlight-debug-time">${formatTime(h.time)}</span>
                <span class="highlight-debug-delta ${deltaClass}">${deltaStr}</span>
                <span class="highlight-debug-type">${h.type || 'event'} ${scoreStr}</span>
                <span class="highlight-debug-game">Game ${h.game_id} ${eventIdStr}</span>
            </div>
            ${valuesStr ? `<div class="highlight-debug-values">[${valuesStr}]</div>` : ''}
        `;
    }).join('');

    debugEl.innerHTML = `
        <h4>${playerIcon}${esc(playerName)} - <span class="good-prob">${playerHighlightCount}</span> / <span class="bad-prob">${playerLowlightCount}</span></h4>
        ${itemsHtml}
    `;

    // Add click handlers
    debugEl.querySelectorAll<HTMLElement>('.highlight-debug-item').forEach(el => {
        el.addEventListener('click', () => {
            const idx = parseInt(el.dataset.highlightIndex!);
            lastHighlightIndex = idx;
            updateDebugActiveHighlight();
            seekTo(playerHighlights[idx].time - HIGHLIGHT_SEEK_BUFFER);
            if (player) player.playVideo();
        });
    });
}

// Update which debug item is marked as active
function updateDebugActiveHighlight(): void {
    const debugEl = document.getElementById('highlightDebug')!;
    debugEl.querySelectorAll<HTMLElement>('.highlight-debug-item').forEach(el => {
        const idx = parseInt(el.dataset.highlightIndex!);
        el.classList.toggle('active', idx === lastHighlightIndex);
    });
    // Scroll active item into view
    const activeEl = debugEl.querySelector('.highlight-debug-item.active');
    if (activeEl) {
        activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function nextHighlight(): void {
    if (playerHighlights.length === 0) return;

    const currentTime = getCurrentTime();
    let nextIndex;

    // If near last highlight, go to next
    if (lastHighlightIndex >= 0 && lastHighlightIndex < playerHighlights.length - 1) {
        const lastTime = playerHighlights[lastHighlightIndex].time;
        if (Math.abs(currentTime - (lastTime - HIGHLIGHT_SEEK_BUFFER)) < 3) {
            nextIndex = lastHighlightIndex + 1;
        }
    }

    // Otherwise find based on current time
    if (nextIndex === undefined) {
        for (let i = 0; i < playerHighlights.length; i++) {
            if (playerHighlights[i].time > currentTime + 0.5) {
                nextIndex = i;
                break;
            }
        }
    }

    if (nextIndex !== undefined && nextIndex < playerHighlights.length) {
        lastHighlightIndex = nextIndex;
        updateDebugActiveHighlight();
        seekTo(playerHighlights[nextIndex].time - HIGHLIGHT_SEEK_BUFFER);
        if (player) player.playVideo();
    }
}

function prevHighlight(): void {
    if (playerHighlights.length === 0) return;

    const currentTime = getCurrentTime();

    // If near last highlight, replay or go to previous
    if (lastHighlightIndex >= 0) {
        const lastTime = playerHighlights[lastHighlightIndex].time;

        // If past the event, replay it
        if (currentTime > lastTime - 0.5) {
            seekTo(lastTime - HIGHLIGHT_SEEK_BUFFER);
            if (player) player.playVideo();
            return;
        }

        // If near the start, go to previous
        if (Math.abs(currentTime - (lastTime - HIGHLIGHT_SEEK_BUFFER)) < 2 && lastHighlightIndex > 0) {
            lastHighlightIndex--;
            updateDebugActiveHighlight();
            seekTo(playerHighlights[lastHighlightIndex].time - HIGHLIGHT_SEEK_BUFFER);
            if (player) player.playVideo();
            return;
        }
    }

    // Otherwise find based on current time
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

// Toggle highlight auto-play mode
function toggleHighlightMode(): void {
    highlightModeEnabled = !highlightModeEnabled;

    // Update button styling
    const btn = document.getElementById('highlightModeBtn')!;
    const mBtn = document.getElementById('mHighlightMode')!;
    btn.classList.toggle('highlight-mode-active', highlightModeEnabled);
    mBtn.classList.toggle('highlight-mode-active', highlightModeEnabled);
    btn.textContent = highlightModeEnabled ? 'Stop' : 'Auto HL';
    mBtn.textContent = highlightModeEnabled ? 'Stop' : 'Auto';

    if (highlightModeEnabled) {
        // Start from first highlight or current position
        if (playerHighlights.length === 0) {
            highlightModeEnabled = false;
            btn.classList.remove('highlight-mode-active');
            mBtn.classList.remove('highlight-mode-active');
            btn.textContent = 'Auto HL';
            mBtn.textContent = 'Auto';
            return;
        }

        // Find the next highlight from current time
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

// Check if we should auto-advance to next highlight
function checkHighlightAutoAdvance(): void {
    if (!highlightModeEnabled || playerHighlights.length === 0) return;
    if (lastHighlightIndex < 0) return;

    const currentTime = getCurrentTime();
    const currentHighlight = playerHighlights[lastHighlightIndex];

    // Check if we've played past the highlight by HIGHLIGHT_PLAY_DURATION seconds
    if (currentTime >= currentHighlight.time + HIGHLIGHT_PLAY_DURATION) {
        // Move to next highlight
        if (lastHighlightIndex < playerHighlights.length - 1) {
            lastHighlightIndex++;
            updateDebugActiveHighlight();
            seekTo(playerHighlights[lastHighlightIndex].time - HIGHLIGHT_SEEK_BUFFER);
        } else {
            // End of highlights - disable mode
            toggleHighlightMode();
        }
    }
}

// Load chapters from JSON
function loadChaptersFromJSON(data: ChapterData): void {
    chapterData = data;
    chapters = data.chapters || [];
    users = data.users || {};
    videoId = data.video_id || null;

    // Build flat list of queen kills from all chapters
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
    // Sort by time (should already be sorted, but ensure it)
    queenKills.sort((a, b) => a.time - b.time);

    // Populate player dropdowns (desktop and mobile)
    const playerSelect = document.getElementById('playerSelect') as HTMLSelectElement;
    const mobilePlayerSelect = document.getElementById('mobilePlayerSelect') as HTMLSelectElement;
    const sortedUsers = Object.entries(users)
        .sort((a, b) => a[1].name.toLowerCase().localeCompare(b[1].name.toLowerCase()));

    if (sortedUsers.length === 0) {
        // No users - replace dropdowns with message
        const noUsersMsg = document.createElement('span');
        noUsersMsg.style.color = '#888';
        noUsersMsg.style.fontStyle = 'italic';
        noUsersMsg.textContent = 'No logged in users';
        playerSelect.parentNode!.replaceChild(noUsersMsg, playerSelect);

        // Hide mobile player selector
        mobilePlayerSelect.style.display = 'none';
    } else {
        playerSelect.innerHTML = '<option value="">Select player...</option>';
        mobilePlayerSelect.innerHTML = '<option value="">Player...</option>';
        for (const [userId, userInfo] of sortedUsers) {
            const option = document.createElement('option');
            option.value = userId;
            option.textContent = userInfo.name;
            playerSelect.appendChild(option);
            // Clone for mobile
            mobilePlayerSelect.appendChild(option.cloneNode(true));
        }

        // Check URL for player param and select that player
        const urlParams = new URLSearchParams(window.location.search);
        const playerParam = urlParams.get('player');
        if (playerParam && users[playerParam]) {
            playerSelect.value = playerParam;
            mobilePlayerSelect.value = playerParam;
            handlePlayerSelect(playerParam, false);  // false = don't update URL again
        }
    }

    renderChapters();
    updateCurrentChapter();
    console.log(`Loaded ${chapters.length} chapters with ${queenKills.length} queen kills and ${Object.keys(users).length} users`);
    if (chapters.length > 0) {
        console.log(`First chapter at ${chapters[0].start_time}s, last at ${chapters[chapters.length-1].start_time}s`);
    }

    // Initialize highlights (works even without position selection)
    updatePlayerHighlights();

    // Initialize YouTube player now that we have the video ID
    initializePlayer();
}

// Cycle team perspective toggle: Auto -> Blue -> Gold -> Auto
function cycleTeamToggle(): void {
    if (favoriteTeam === null) favoriteTeam = 'blue';
    else if (favoriteTeam === 'blue') favoriteTeam = 'gold';
    else favoriteTeam = null;

    const btn = document.getElementById('teamToggle')!;
    if (favoriteTeam === null) btn.textContent = 'Team: Auto';
    else if (favoriteTeam === 'blue') btn.textContent = 'Team: Blue';
    else btn.textContent = 'Team: Gold';

    // Force UI refresh
    renderChapters(chapterFilter.value);
    if (player && player.getCurrentTime) {
        const t = player.getCurrentTime();
        updateContributionBars(t);
        updateEggGrid(t);
        updateBerryGrid(t);
        updateOverlay(t);
    }
}

// Event listeners
document.getElementById('prevChapter')!.addEventListener('click', prevChapter);
document.getElementById('nextChapter')!.addEventListener('click', nextChapter);
document.getElementById('prevSet')!.addEventListener('click', prevSet);
document.getElementById('nextSet')!.addEventListener('click', nextSet);
document.getElementById('prevEgg')!.addEventListener('click', prevQueenKill);
document.getElementById('nextEgg')!.addEventListener('click', nextQueenKill);
document.getElementById('prevHighlightBtn')!.addEventListener('click', prevHighlight);
document.getElementById('nextHighlightBtn')!.addEventListener('click', nextHighlight);
document.getElementById('highlightModeBtn')!.addEventListener('click', toggleHighlightMode);
document.getElementById('teamToggle')!.addEventListener('click', cycleTeamToggle);
playPauseBtn.addEventListener('click', togglePlayPause);

chapterFilter.addEventListener('input', (e) => {
    renderChapters((e.target as HTMLInputElement).value);
});

// Helper: get user's position in a chapter
function getUserPositionInChapter(userId: string, chapter: Chapter): number | null {
    if (!chapter.users) return null;
    for (const [pos, uid] of Object.entries(chapter.users)) {
        if (String(uid) === String(userId)) {
            return parseInt(pos);
        }
    }
    return null;
}

// Player selector (by name) - shared handler for desktop and mobile
function handlePlayerSelect(userId: string, updateUrl: boolean = true): void {
    selectedUserId = userId;
    // Reset team toggle to auto when selecting a player
    favoriteTeam = null;
    document.getElementById('teamToggle')!.textContent = 'Team: Auto';
    // Clear position selectors when using player selector
    (document.getElementById('positionSelect') as HTMLSelectElement).value = '';
    (document.getElementById('mobilePositionSelect') as HTMLSelectElement).value = '';

    if (selectedUserId && currentChapterIndex >= 0) {
        // Find this user's position in the current chapter
        const pos = getUserPositionInChapter(selectedUserId, chapters[currentChapterIndex]);
        selectedPosition = pos ? String(pos) : null;
    } else {
        selectedPosition = null;
    }

    // Update URL with player param
    if (updateUrl) {
        const url = new URL(window.location.href);
        if (userId) {
            url.searchParams.set('player', userId);
        } else {
            url.searchParams.delete('player');
        }
        window.history.replaceState({}, '', url);
    }

    updatePlayerHighlights();
    renderChapters(chapterFilter.value);
}

(document.getElementById('playerSelect') as HTMLSelectElement).addEventListener('change', (e) => {
    handlePlayerSelect((e.target as HTMLSelectElement).value);
    // Sync mobile selector
    (document.getElementById('mobilePlayerSelect') as HTMLSelectElement).value = (e.target as HTMLSelectElement).value;
});

(document.getElementById('mobilePlayerSelect') as HTMLSelectElement).addEventListener('change', (e) => {
    handlePlayerSelect((e.target as HTMLSelectElement).value);
    // Sync desktop selector
    (document.getElementById('playerSelect') as HTMLSelectElement).value = (e.target as HTMLSelectElement).value;
});

// Position selector - shared handler for desktop and mobile
function handlePositionSelect(position: string): void {
    selectedPosition = position;
    selectedUserId = null;
    // Reset team toggle to auto when selecting a position
    favoriteTeam = null;
    document.getElementById('teamToggle')!.textContent = 'Team: Auto';
    (document.getElementById('playerSelect') as HTMLSelectElement).value = '';
    (document.getElementById('mobilePlayerSelect') as HTMLSelectElement).value = '';
    (document.getElementById('positionSelect') as HTMLSelectElement).value = position;
    (document.getElementById('mobilePositionSelect') as HTMLSelectElement).value = position;
    updatePlayerHighlights();
    renderChapters(chapterFilter.value);
}

(document.getElementById('positionSelect') as HTMLSelectElement).addEventListener('change', (e) => {
    handlePositionSelect((e.target as HTMLSelectElement).value);
});

// Mobile controls
let mobileControlsVisible = true;

document.getElementById('mobileToggle')!.addEventListener('click', () => {
    mobileControlsVisible = !mobileControlsVisible;
    document.getElementById('mobileControls')!.classList.toggle('visible', mobileControlsVisible);
    document.getElementById('mobileToggle')!.textContent = mobileControlsVisible ? '✕' : '☰';
});

document.getElementById('mPrevSet')!.addEventListener('click', prevSet);
document.getElementById('mNextSet')!.addEventListener('click', nextSet);
document.getElementById('mPrevGame')!.addEventListener('click', prevChapter);
document.getElementById('mNextGame')!.addEventListener('click', nextChapter);
document.getElementById('mPrevHighlight')!.addEventListener('click', prevHighlight);
document.getElementById('mNextHighlight')!.addEventListener('click', nextHighlight);
document.getElementById('mHighlightMode')!.addEventListener('click', toggleHighlightMode);

(document.getElementById('mobilePositionSelect') as HTMLSelectElement).addEventListener('change', (e) => {
    handlePositionSelect((e.target as HTMLSelectElement).value);
});

// File loading (hidden input, kept for flexibility)
(document.getElementById('chaptersInput') as HTMLInputElement).addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files![0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const data = JSON.parse(event.target!.result as string);
                loadChaptersFromJSON(data);
            } catch (err) {
                alert('Error parsing chapters.json: ' + (err as Error).message);
            }
        };
        reader.readAsText(file);
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ignore if typing in input
    if ((e.target as HTMLElement).tagName === 'INPUT') return;

    switch (e.key) {
        case ' ':
            e.preventDefault();
            togglePlayPause();
            break;
        case 'ArrowLeft':
            seekTo(getCurrentTime() - 5);
            break;
        case 'ArrowRight':
            seekTo(getCurrentTime() + 5);
            break;
        case 'j':
            seekTo(getCurrentTime() - 10);
            break;
        case 'l':
            seekTo(getCurrentTime() + 10);
            break;
        case 'p':
            prevChapter();
            break;
        case 'n':
            nextChapter();
            break;
        case 'g':
            nextChapter();
            break;
        case 'G':
            prevChapter();
            break;
        case 's':
            nextSet();
            break;
        case 'S':
            prevSet();
            break;
        case 'e':
            nextQueenKill();
            break;
        case 'E':
            prevQueenKill();
            break;
        case 'h':
            nextHighlight();
            break;
        case 'H':
            prevHighlight();
            break;
        case 'a':
        case 'A':
            toggleHighlightMode();
            break;
        case 't':
        case 'T':
            cycleTeamToggle();
            break;
        case 'x':
        case 'X':
            currentCellStyle = currentCellStyle === 'pulse' ? 'bold' : 'pulse';
            { const t = getCurrentTime(); updateEggGrid(t); updateBerryGrid(t); }
            break;
        case '1': case '2': case '3': case '4': case '5':
        case '6': case '7': case '8': case '9': case '0':
            // Map keyboard to internal position IDs (viewer perspective)
            const keyToPosition: Record<string, string> = {
                '1': '10',  // Blue Checkers
                '2': '8',   // Blue Skull
                '3': '2',   // Blue Queen
                '4': '9',   // Blue Abs
                '5': '7',   // Blue Stripes
                '6': '6',   // Gold Checkers
                '7': '4',   // Gold Skull
                '8': '1',   // Gold Queen
                '9': '5',   // Gold Abs
                '0': '3',   // Gold Stripes
            };
            const posSelect = document.getElementById('positionSelect') as HTMLSelectElement;
            posSelect.value = keyToPosition[e.key];
            selectedPosition = keyToPosition[e.key];
            updatePlayerHighlights();
            renderChapters(chapterFilter.value);  // Re-render to show player dots on plots
            // Sync mobile selector
            (document.getElementById('mobilePositionSelect') as HTMLSelectElement).value = selectedPosition;
            break;
    }
});

// --- Calibration mode (landmark-based: click on speed gates) ---
let calibrationClicks: CalibrationClick[] = [];
let calibrating = false;
let calibrationGates: SpeedGates | null = null;  // {left: {gx, gy}, right: {gx, gy}} in effective (flipped) coords

// Find speed gates for the current chapter, returning effective screen coords
function getSpeedGates(ch: Chapter | null): SpeedGates | null {
    if (!ch || !ch.map) return null;
    const mapInfo = MAP_STRUCTURE[ch.map];
    if (!mapInfo) return null;

    const speedGates = mapInfo.maiden_info
        .filter(m => m[0] === 'maiden_speed')
        .map(m => ({ gx: m[1], gy: m[2] }));
    if (speedGates.length < 2) return null;

    // Apply flip if gold_on_left is false (screen is mirrored)
    const needsFlip = !ch.gold_on_left;
    const effectiveGates = speedGates.map(g => ({
        gx: needsFlip ? (1920 - g.gx) : g.gx,
        gy: g.gy
    }));

    // Sort by effective X so left/right matches what user sees on screen
    effectiveGates.sort((a, b) => a.gx - b.gx);
    return { left: effectiveGates[0], right: effectiveGates[effectiveGates.length - 1] };
}

document.getElementById('calibrateBtn')!.addEventListener('click', () => {
    calibrating = !calibrating;
    calibrationClicks = [];
    calibrationGates = null;
    const btn = document.getElementById('calibrateBtn')!;
    const overlay = document.getElementById('cfOverlay')!;

    if (calibrating) {
        const ch = currentChapterIndex >= 0 ? chapters[currentChapterIndex] : null;
        calibrationGates = getSpeedGates(ch);

        if (!calibrationGates) {
            btn.textContent = 'No speed gates found for current map';
            btn.style.background = '#f44336';
            calibrating = false;
            setTimeout(() => { btn.textContent = 'Calibrate'; btn.style.background = ''; }, 2000);
            return;
        }

        btn.textContent = 'Click on LEFT speed gate';
        btn.style.background = '#e94560';
        overlay.style.pointerEvents = 'auto';
        overlay.style.cursor = 'crosshair';
        overlay.style.background = 'rgba(0,0,0,0.15)';
        overlay.innerHTML = '';
    } else {
        btn.textContent = 'Calibrate';
        btn.style.background = '';
        overlay.style.pointerEvents = 'none';
        overlay.style.cursor = '';
        overlay.style.background = '';
    }
});

document.getElementById('cfOverlay')!.addEventListener('click', (e: MouseEvent) => {
    if (!calibrating) return;
    e.stopPropagation();
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const xPct = (e.clientX - rect.left) / rect.width * 100;
    const yPct = (e.clientY - rect.top) / rect.height * 100;
    calibrationClicks.push({ x: xPct, y: yPct });
    const btn = document.getElementById('calibrateBtn')!;
    const overlay = document.getElementById('cfOverlay')!;

    function crosshairHTML(x: number, y: number): string {
        return `<div style="position:absolute;left:${x}%;top:0;width:2px;height:100%;background:rgba(255,255,0,0.7);pointer-events:none;"></div>
            <div style="position:absolute;top:${y}%;left:0;width:100%;height:2px;background:rgba(255,255,0,0.7);pointer-events:none;"></div>
            <div style="position:absolute;left:${x}%;top:${y}%;width:16px;height:16px;border-radius:50%;background:yellow;border:2px solid red;transform:translate(-50%,-50%);pointer-events:none;z-index:999;"></div>`;
    }

    if (calibrationClicks.length === 1) {
        overlay.innerHTML = crosshairHTML(xPct, yPct);
        btn.textContent = 'Click on RIGHT speed gate';
    } else if (calibrationClicks.length === 2) {
        const ox1 = calibrationClicks[0].x, oy1 = calibrationClicks[0].y;
        const ox2 = calibrationClicks[1].x, oy2 = calibrationClicks[1].y;
        const gx1 = calibrationGates!.left.gx, gy1 = calibrationGates!.left.gy;
        const gx2 = calibrationGates!.right.gx, gy2 = calibrationGates!.right.gy;

        // Solve affine transform: overlayX% = a_x + b_x * gameX
        const b_x = (ox2 - ox1) / (gx2 - gx1);
        const a_x = ox1 - b_x * gx1;
        // Uniform scaling: overlay is 16:9, so 1% width = (16/9) * 1% height in pixels
        const b_y = b_x * 16 / 9;
        // Game coords are y-up; toPercent uses (1080 - y), so solve with flipped y
        const a_y = oy1 - b_y * (1080 - gy1);

        const gameTransform = {
            a_x: Math.round(a_x * 1000) / 1000,
            b_x: Math.round(b_x * 100000) / 100000,
            a_y: Math.round(a_y * 1000) / 1000,
            b_y: Math.round(b_y * 100000) / 100000
        };

        if (chapterData) {
            chapterData.game_transform = gameTransform;
        }
        console.log('game_transform:', JSON.stringify(gameTransform));
        const json = JSON.stringify(gameTransform);
        navigator.clipboard.writeText(`"game_transform": ${json}`).then(() => {
            btn.textContent = 'Calibrated! (copied to clipboard)';
        }).catch(() => {
            btn.textContent = 'Calibrated! Check console';
        });
        btn.style.background = '#4caf50';
        calibrating = false;
        calibrationClicks = [];
        calibrationGates = null;
        overlay.style.pointerEvents = 'none';
        overlay.style.cursor = '';
        overlay.style.background = '';
        if (player && player.getCurrentTime) {
            updateOverlay(player.getCurrentTime());
        }
    }
});

// Get chapters URL from query parameter or use default
const urlParams = new URLSearchParams(window.location.search);
const chaptersUrl = urlParams.get('chapters') || 'chapters/tournaments/842.json';

// Auto-load chapters file
fetch(chaptersUrl)
    .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
    })
    .then(loadChaptersFromJSON)
    .catch((err) => console.log(`Failed to load ${chaptersUrl}: ${err.message}`));
