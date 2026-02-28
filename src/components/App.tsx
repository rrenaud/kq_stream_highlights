import { useEffect, useRef, useState, useCallback, useMemo } from 'preact/hooks';
import type { ChapterData } from '../types';
import { fetchJSON } from '../fetch';
import { HIGHLIGHT_SEEK_BUFFER } from '../constants';
import { formatTime, getUserPositionInChapter, perspectiveDelta } from '../utils';
import { findChapterAtTime, jumpToChapter, prevChapter, nextChapter, prevSet, nextSet, nextQueenKill, prevQueenKill } from '../navigation';
import { computePlayerHighlights, nextHighlight, prevHighlight, checkHighlightAutoAdvance } from '../highlights';
import { setupYouTubeAPI, initializePlayer } from '../youtube';
import { getCurrentTime, seekTo, togglePlayPause } from '../state';
import { createCalibrationState, getSpeedGates, handleCalibrationClick } from '../calibration';
import {
    ytPlayer, chapters, queenKills, currentChapterIndex, currentTime,
    favoriteTeam, selectedPosition, selectedUserId,
    playerHighlights, playerHighlightCount, playerLowlightCount, lastHighlightIndex, highlightModeEnabled,
    users, videoId, chapterData, currentChapter, flipForGold,
    videos as videosSignal, currentVideoSource, cabFilter,
} from '../state';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';

import { MapOverlay } from './MapOverlay';
import { EggGrid } from './EggGrid';
import { BerryGrid } from './BerryGrid';
import { ContributionBars } from './ContributionBars';
import { ChapterList } from './ChapterList';
import { HighlightDebug } from './HighlightDebug';
import { KeyboardHints } from './KeyboardHints';

export function App() {
    const calState = useMemo(() => createCalibrationState(), []);

    const [filter, setFilter] = useState('');
    const [mobileControlsVisible, setMobileControlsVisible] = useState(true);
    const [playPauseText, setPlayPauseText] = useState('\u25B6 Play');
    const [calibrateText, setCalibrateText] = useState('Calibrate');
    const [calibrateBg, setCalibrateBg] = useState('');
    const [calibrating, setCalibrating] = useState(false);

    // Force re-render when signals change (read signal values to subscribe)
    const _ct = currentTime.value;
    const _ci = currentChapterIndex.value;
    const _ch = currentChapter.value;
    const _ffg = flipForGold.value;
    const _cd = chapterData.value;
    const _chs = chapters.value;
    const _sp = selectedPosition.value;
    const _su = selectedUserId.value;
    const _ft = favoriteTeam.value;
    const _ph = playerHighlights.value;
    const _phc = playerHighlightCount.value;
    const _plc = playerLowlightCount.value;
    const _lhi = lastHighlightIndex.value;
    const _hme = highlightModeEnabled.value;
    const _us = users.value;
    const _vs = videosSignal.value;
    const _cabFilter = cabFilter.value;

    // Derived values (replace useState that was manually synced)
    const teamToggleText = _ft === null ? 'Team: Auto' : _ft === 'blue' ? 'Team: Blue' : 'Team: Gold';
    const highlightBtnText = _hme ? 'Stop' : 'Auto HL';
    const mHighlightBtnText = _hme ? 'Stop' : 'Auto';

    const overlayRef = useRef<HTMLDivElement>(null);
    const initialized = useRef(false);

    // --- Load chapters from JSON data ---
    const loadChaptersFromJSON = useCallback((data: ChapterData) => {
        chapterData.value = data;
        chapters.value = data.chapters || [];
        users.value = data.users || {};
        videoId.value = data.video_id || null;

        // Multi-video support
        videosSignal.value = data.videos || {};
        const vKeys = Object.keys(data.videos || {});
        if (vKeys.length > 0) {
            const firstKey = vKeys[0];
            currentVideoSource.value = firstKey;
            if (!data.video_id && data.videos![firstKey].video_id) {
                videoId.value = data.videos![firstKey].video_id;
            }
        }

        const flatKills: typeof queenKills.value = [];
        for (const ch of (data.chapters || [])) {
            if (ch.queen_kills) {
                for (const qk of ch.queen_kills) {
                    flatKills.push({ time: qk.time, victim: qk.victim, game_id: ch.game_id });
                }
            }
        }
        flatKills.sort((a, b) => a.time - b.time);
        queenKills.value = flatKills;

        currentChapterIndex.value = -1;
        lastHighlightIndex.value = -1;

        // Handle ?player= URL param
        const urlParams = new URLSearchParams(window.location.search);
        const playerParam = urlParams.get('player');
        if (playerParam && data.users && data.users[playerParam]) {
            selectedUserId.value = playerParam;
            favoriteTeam.value = null;
            selectedPosition.value = null;
        }

        // Compute highlights
        recomputeHighlights();

        console.log(`Loaded ${(data.chapters || []).length} chapters with ${flatKills.length} queen kills and ${Object.keys(data.users || {}).length} users`);

        // Initialize YouTube player if API is ready
        initializePlayer({
            onReady: onPlayerReady,
            onStateChange: onPlayerStateChange,
        });
    }, []);

    // --- YouTube callbacks ---
    function onPlayerReady(_event: YT.PlayerEvent) {
        console.log('YouTube player ready');

        const params = new URLSearchParams(window.location.search);
        const gameParam = params.get('game');
        const tParam = params.get('t');
        if (gameParam) {
            const gid = parseInt(gameParam);
            const idx = chapters.value.findIndex(ch => ch.game_id === gid);
            if (idx >= 0) {
                const seekTime = tParam ? chapters.value[idx].start_time + parseFloat(tParam) : chapters.value[idx].start_time;
                seekTo(seekTime);
                console.log(`URL nav: game ${gid} (chapter ${idx}), seeking to ${seekTime}s`);
            }
        } else if (tParam) {
            seekTo(parseFloat(tParam));
        }
    }

    function onPlayerStateChange(event: YT.OnStateChangeEvent) {
        if (event.data === YT.PlayerState.PLAYING) {
            setPlayPauseText('\u23F8 Pause');
        } else {
            setPlayPauseText('\u25B6 Play');
        }
    }

    // --- YouTube setup + data loading (once on mount) ---
    useEffect(() => {
        if (initialized.current) return;
        initialized.current = true;

        setupYouTubeAPI({ onReady: onPlayerReady, onStateChange: onPlayerStateChange });

        const urlParams = new URLSearchParams(window.location.search);
        const chaptersUrl = urlParams.get('chapters') || 'chapters/tournaments/842.json';
        fetchJSON<ChapterData>(chaptersUrl)
            .then(loadChaptersFromJSON)
            .catch(err => console.log(`Failed to load ${chaptersUrl}: ${err.message}`));
    }, [loadChaptersFromJSON]);

    // --- 500ms polling interval for time + chapter tracking ---
    useEffect(() => {
        const interval = setInterval(() => {
            const player = ytPlayer.value;
            if (!player || !player.getCurrentTime) return;

            const t = player.getCurrentTime();
            currentTime.value = t;

            // Update chapter index
            const newIndex = findChapterAtTime(chapters.value, t);
            if (newIndex !== currentChapterIndex.value) {
                currentChapterIndex.value = newIndex;

                if (selectedUserId.value && newIndex >= 0) {
                    const pos = getUserPositionInChapter(selectedUserId.value, chapters.value[newIndex]);
                    selectedPosition.value = pos ? String(pos) : null;
                }
            }

            // Check highlight auto-advance
            checkHighlightAutoAdvance(toggleHighlightMode);
        }, 500);

        return () => clearInterval(interval);
    }, []);

    // --- Auto-scroll active chapter ---
    useEffect(() => {
        const activeEl = document.querySelector('.chapter-item.active');
        if (activeEl) {
            activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }, [_ci]);

    // --- Highlight computation ---
    function recomputeHighlights() {
        const highlights = computePlayerHighlights();
        playerHighlights.value = highlights;
        lastHighlightIndex.value = -1;

        let hCount = 0, lCount = 0;
        for (const h of highlights) {
            if (perspectiveDelta(h.delta, h.position) >= 0) hCount++;
            else lCount++;
        }
        playerHighlightCount.value = hCount;
        playerLowlightCount.value = lCount;
    }

    // Recompute highlights when selection changes
    useEffect(() => {
        if (chapters.value.length > 0) {
            recomputeHighlights();
        }
    }, [_su, _sp, _chs]);

    // --- Highlight mode toggle ---
    function toggleHighlightMode() {
        const newEnabled = !highlightModeEnabled.value;
        highlightModeEnabled.value = newEnabled;

        if (newEnabled) {
            if (playerHighlights.value.length === 0) {
                highlightModeEnabled.value = false;
                return;
            }

            const ct = getCurrentTime();
            let startIdx = 0;
            for (let i = 0; i < playerHighlights.value.length; i++) {
                if (playerHighlights.value[i].time > ct) {
                    startIdx = i;
                    break;
                }
            }

            lastHighlightIndex.value = startIdx;
            seekTo(playerHighlights.value[startIdx].time - HIGHLIGHT_SEEK_BUFFER);
            if (ytPlayer.value) ytPlayer.value.playVideo();
        }
    }

    // --- Team toggle ---
    function cycleTeamToggle() {
        if (favoriteTeam.value === null) favoriteTeam.value = 'blue';
        else if (favoriteTeam.value === 'blue') favoriteTeam.value = 'gold';
        else favoriteTeam.value = null;
    }

    // --- Player/Position selection ---
    function handlePlayerSelect(userId: string, updateUrl: boolean = true) {
        selectedUserId.value = userId || null;
        favoriteTeam.value = null;
        selectedPosition.value = null;

        if (userId && currentChapterIndex.value >= 0) {
            const pos = getUserPositionInChapter(userId, chapters.value[currentChapterIndex.value]);
            selectedPosition.value = pos ? String(pos) : null;
        }

        if (updateUrl) {
            const url = new URL(window.location.href);
            if (userId) url.searchParams.set('player', userId);
            else url.searchParams.delete('player');
            window.history.replaceState({}, '', url);
        }
    }

    function handlePositionSelect(position: string) {
        selectedPosition.value = position || null;
        selectedUserId.value = null;
        favoriteTeam.value = null;
    }

    // --- Navigation callbacks ---
    function handleJumpToChapter(index: number) {
        jumpToChapter(index);
    }

    function handleSeek(time: number) {
        seekTo(time);
        if (ytPlayer.value) ytPlayer.value.playVideo();
    }

    function handleHighlightClick(index: number) {
        lastHighlightIndex.value = index;
        seekTo(playerHighlights.value[index].time - HIGHLIGHT_SEEK_BUFFER);
        if (ytPlayer.value) ytPlayer.value.playVideo();
    }

    // --- Calibration ---
    function handleCalibrateClick() {
        const nowCalibrating = !calibrating;
        calState.clicks = [];
        calState.gates = null;
        const overlay = overlayRef.current;

        if (nowCalibrating) {
            const ch = currentChapterIndex.value >= 0 ? chapters.value[currentChapterIndex.value] : null;
            calState.gates = getSpeedGates(ch);

            if (!calState.gates) {
                setCalibrateText('No speed gates found for current map');
                setCalibrateBg('#f44336');
                setCalibrating(false);
                setTimeout(() => { setCalibrateText('Calibrate'); setCalibrateBg(''); }, 2000);
                return;
            }

            setCalibrateText('Click on LEFT speed gate');
            setCalibrateBg('#e94560');
            setCalibrating(true);
            if (overlay) {
                overlay.style.pointerEvents = 'auto';
                overlay.style.cursor = 'crosshair';
                overlay.style.background = 'rgba(0,0,0,0.15)';
                overlay.innerHTML = '';
            }
        } else {
            setCalibrateText('Calibrate');
            setCalibrateBg('');
            setCalibrating(false);
            if (overlay) {
                overlay.style.pointerEvents = 'none';
                overlay.style.cursor = '';
                overlay.style.background = '';
            }
        }
    }

    function handleOverlayClick(e: MouseEvent) {
        if (!calibrating) return;
        e.stopPropagation();
        const overlay = overlayRef.current;
        if (!overlay) return;
        const rect = overlay.getBoundingClientRect();
        const xPct = (e.clientX - rect.left) / rect.width * 100;
        const yPct = (e.clientY - rect.top) / rect.height * 100;

        // We need a fake btn element for the calibration callback
        const btnProxy = {
            get textContent() { return calibrateText; },
            set textContent(v: string) { setCalibrateText(v); },
            style: { get background() { return calibrateBg; }, set background(v: string) { setCalibrateBg(v); } },
        } as unknown as HTMLElement;

        handleCalibrationClick(calState, xPct, yPct, btnProxy, overlay, () => {
            setCalibrating(false);
        });
    }

    // --- Keyboard shortcuts ---
    useKeyboardShortcuts({ toggleHighlightMode, cycleTeamToggle, handlePositionSelect });

    // --- Derived values for rendering ---
    const timeDisplayText = `${formatTime(_ct)} / ${formatTime(ytPlayer.value?.getDuration?.() || 0)}`;

    const sortedUsers = useMemo(() => {
        return Object.entries(_us)
            .sort((a, b) => a[1].name.toLowerCase().localeCompare(b[1].name.toLowerCase()));
    }, [_us]);

    return (
        <>
            <div class="container">
                <div class="video-section">
                    <div class="video-wrapper">
                        <div id="player"></div>
                        {calibrating ? (
                            <div
                                ref={overlayRef}
                                id="cfOverlay"
                                class="cf-overlay"
                                style="pointer-events:auto;cursor:crosshair;background:rgba(0,0,0,0.15);"
                                onClick={handleOverlayClick}
                            ></div>
                        ) : (
                            <MapOverlay
                                ch={_ch}
                                currentTime={_ct}
                                flipForGold={_ffg}
                                chapterData={_cd}
                            />
                        )}
                    </div>

                    <div class="controls">
                        <button onClick={() => togglePlayPause()}>{playPauseText}</button>
                        <span class="time-display">{timeDisplayText}</span>
                        <span class="nav-group">
                            <button onClick={() => prevSet()} title="Previous set (Shift+S)">{'\u25C0'}Set</button>
                            <button onClick={() => nextSet()} title="Next set (S)">Set{'\u25B6'}</button>
                        </span>
                        <span class="nav-group">
                            <button onClick={() => prevChapter()} title="Previous game (Shift+G)">{'\u25C0'}Game</button>
                            <button onClick={() => nextChapter()} title="Next game (G)">Game{'\u25B6'}</button>
                        </span>
                        <span class="nav-group">
                            <button onClick={() => prevQueenKill()} title="Previous queen kill (Shift+E)">{'\u25C0'}Egg</button>
                            <button onClick={() => nextQueenKill()} title="Next queen kill (E)">Egg{'\u25B6'}</button>
                        </span>
                        <span class="nav-group">
                            <button onClick={() => prevHighlight()} title="Previous highlight (Shift+H)">{'\u25C0'}HL</button>
                            <button onClick={() => nextHighlight()} title="Next highlight (H)">HL{'\u25B6'}</button>
                        </span>
                        <button
                            class={_hme ? 'highlight-mode-active' : ''}
                            onClick={toggleHighlightMode}
                            title="Auto-play highlights (A)"
                        >{highlightBtnText}</button>
                        <button onClick={cycleTeamToggle} title="Toggle team perspective (T)">{teamToggleText}</button>
                        <button
                            onClick={handleCalibrateClick}
                            title="Click to calibrate game rectangle"
                            style={calibrateBg ? `background:${calibrateBg}` : ''}
                        >{calibrateText}</button>
                        <span class="current-chapter">
                            {_ch && (
                                <>
                                    <h3>{_ch.title}</h3>
                                    <span class={_ch.winner}>{_ch.winner}</span> wins by {_ch.win_condition}
                                    &nbsp;|&nbsp; {formatTime(_ch.duration)}
                                    &nbsp;|&nbsp; <a href={_ch.hivemind_url} target="_blank" style="color: #e94560;">HiveMind</a>
                                </>
                            )}
                        </span>
                    </div>

                    <div class="cf-grids-row">
                        <EggGrid ch={_ch} currentTime={_ct} flipForGold={_ffg} />
                        <BerryGrid ch={_ch} currentTime={_ct} flipForGold={_ffg} />
                    </div>

                    <ContributionBars ch={_ch} currentTime={_ct} flipForGold={_ffg} chapterData={_cd} />

                    <KeyboardHints />

                    <HighlightDebug
                        highlights={_ph}
                        highlightCount={_phc}
                        lowlightCount={_plc}
                        lastHighlightIndex={_lhi}
                        selectedUserId={_su}
                        selectedPosition={_sp}
                        users={_us}
                        currentChapterIndex={_ci}
                        chapters={_chs}
                        onHighlightClick={handleHighlightClick}
                    />
                </div>

                <div class="chapter-sidebar">
                    <div class="sidebar-header">
                        <h2>Chapters</h2>
                        <input
                            type="text"
                            placeholder="Filter by map, winner..."
                            value={filter}
                            onInput={(e) => setFilter((e.target as HTMLInputElement).value)}
                        />
                        {Object.keys(_vs).length > 1 && (
                            <div class="cab-filter">
                                <button
                                    class={`cab-filter-btn${!_cabFilter ? ' active' : ''}`}
                                    onClick={() => { cabFilter.value = null; }}
                                >All</button>
                                {Object.entries(_vs).map(([key, v]) => (
                                    <button
                                        key={key}
                                        class={`cab-filter-btn${_cabFilter === key ? ' active' : ''}`}
                                        onClick={() => { cabFilter.value = _cabFilter === key ? null : key; }}
                                    >{v.label}</button>
                                ))}
                            </div>
                        )}
                        <div class="position-selector">
                            <label>Player:</label>
                            {sortedUsers.length === 0 ? (
                                <span style="color:#888;font-style:italic;">No logged in users</span>
                            ) : (
                                <select
                                    value={_su || ''}
                                    onChange={(e) => {
                                        handlePlayerSelect((e.target as HTMLSelectElement).value);
                                    }}
                                >
                                    <option value="">Select player...</option>
                                    {sortedUsers.map(([uid, info]) => (
                                        <option key={uid} value={uid}>{info.name}</option>
                                    ))}
                                </select>
                            )}
                        </div>
                        <div class="position-selector">
                            <label>Position:</label>
                            <select
                                value={_sp || ''}
                                onChange={(e) => handlePositionSelect((e.target as HTMLSelectElement).value)}
                            >
                                <option value="">Or by position...</option>
                                <option value="10">1 - {'\u25A6'} Blue Checkers</option>
                                <option value="8">2 - {'\uD83D\uDC80'} Blue Skull</option>
                                <option value="2">3 - {'\uD83D\uDC51'} Blue Queen</option>
                                <option value="9">4 - {'\uD83D\uDCAA'} Blue Abs</option>
                                <option value="7">5 - {'\u2630'} Blue Stripes</option>
                                <option value="6">6 - {'\u25A6'} Gold Checkers</option>
                                <option value="4">7 - {'\uD83D\uDC80'} Gold Skull</option>
                                <option value="1">8 - {'\uD83D\uDC51'} Gold Queen</option>
                                <option value="5">9 - {'\uD83D\uDCAA'} Gold Abs</option>
                                <option value="3">0 - {'\u2630'} Gold Stripes</option>
                            </select>
                            {(_su || _sp) && (
                                <span>
                                    <span class="good-prob">{_phc}</span> / <span class="bad-prob">{_plc}</span>
                                </span>
                            )}
                        </div>
                    </div>
                    <ChapterList
                        chapters={_chs}
                        currentChapterIndex={_ci}
                        selectedPosition={_sp}
                        selectedUserId={_su}
                        favoriteTeam={_ft}
                        filter={filter}
                        videos={_vs}
                        cabFilter={_cabFilter}
                        onJumpToChapter={handleJumpToChapter}
                        onSeek={handleSeek}
                    />
                </div>
            </div>

            {/* Mobile toggle button */}
            <button
                class="mobile-toggle"
                onClick={() => {
                    setMobileControlsVisible(!mobileControlsVisible);
                }}
            >{mobileControlsVisible ? '\u2715' : '\u2630'}</button>

            {/* Mobile control bar */}
            <div class={`mobile-controls${mobileControlsVisible ? ' visible' : ''}`}>
                <div class="mobile-nav-row">
                    <div class="nav-group-left">
                        <button onClick={() => prevSet()}>{'\u25C0\u25C0'} S</button>
                        <button onClick={() => prevChapter()}>{'\u25C0'} G</button>
                        <button onClick={() => prevHighlight()}>{'\u25C0'} H</button>
                        {sortedUsers.length > 0 && (
                            <select
                                class="mobile-select"
                                value={_su || ''}
                                onChange={(e) => handlePlayerSelect((e.target as HTMLSelectElement).value)}
                            >
                                <option value="">Player...</option>
                                {sortedUsers.map(([uid, info]) => (
                                    <option key={uid} value={uid}>{info.name}</option>
                                ))}
                            </select>
                        )}
                    </div>
                    <div class="nav-group-right">
                        <button onClick={() => nextSet()}>S {'\u25B6\u25B6'}</button>
                        <button onClick={() => nextChapter()}>G {'\u25B6'}</button>
                        <button onClick={() => nextHighlight()}>H {'\u25B6'}</button>
                        <button
                            class={_hme ? 'highlight-mode-active' : ''}
                            onClick={toggleHighlightMode}
                        >{mHighlightBtnText}</button>
                        <select
                            class="mobile-select"
                            value={_sp || ''}
                            onChange={(e) => handlePositionSelect((e.target as HTMLSelectElement).value)}
                        >
                            <option value="">Position...</option>
                            <option value="10">1 {'\u25A6'} Blue Chk</option>
                            <option value="8">2 {'\uD83D\uDC80'} Blue Skl</option>
                            <option value="2">3 {'\uD83D\uDC51'} Blue Qn</option>
                            <option value="9">4 {'\uD83D\uDCAA'} Blue Abs</option>
                            <option value="7">5 {'\u2630'} Blue Str</option>
                            <option value="6">6 {'\u25A6'} Gold Chk</option>
                            <option value="4">7 {'\uD83D\uDC80'} Gold Skl</option>
                            <option value="1">8 {'\uD83D\uDC51'} Gold Qn</option>
                            <option value="5">9 {'\uD83D\uDCAA'} Gold Abs</option>
                            <option value="3">0 {'\u2630'} Gold Str</option>
                        </select>
                    </div>
                </div>
            </div>

        </>
    );
}
