import { getCurrentTime, seekTo, chapters, selectedUserId, selectedPosition, playerHighlights, lastHighlightIndex, highlightModeEnabled, ytPlayer, currentVideoSource } from './state';
import type { Chapter, PlayerHighlight, HighImpactRange, KDStats } from './types';
import { HIGHLIGHT_SEEK_BUFFER, HIGHLIGHT_PLAY_DURATION } from './constants';
import { isGoldTeam, perspectiveDelta, getUserPositionInChapter } from './utils';

export function calculateNetWinProb(ch: Chapter, positionId: string): number | null {
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

export function findHighImpactRanges(ch: Chapter, positionId: string): HighImpactRange[] {
    if (!positionId || !ch.player_events) return [];
    const pos = parseInt(positionId);

    const playerEvents = ch.player_events
        .filter(evt => evt.positions && evt.positions.includes(pos) && Math.abs(evt.delta) >= 0.05)
        .sort((a, b) => a.time - b.time);

    if (playerEvents.length === 0) return [];

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
    if (rangeStart !== null && Math.abs(rangeDelta) >= 0.10) {
        ranges.push({ start: rangeStart, end: rangeEnd!, delta: rangeDelta });
    }

    return ranges;
}

export function calculateKD(ch: Chapter, positionId: string): KDStats | null {
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

export function computePlayerHighlights(): PlayerHighlight[] {
    let allEvents: PlayerHighlight[] = [];
    let anyQueen = false;
    const noSelection = !selectedUserId.value && !selectedPosition.value;

    for (const ch of chapters.value) {
        if (!ch.player_events) continue;

        if (noSelection) {
            for (const evt of ch.player_events) {
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
                        ml_score: evt.ml_score,
                        video_source: ch.video_source,
                    });
                }
            }
        } else {
            let pos;
            if (selectedUserId.value) {
                pos = getUserPositionInChapter(selectedUserId.value, ch);
                if (!pos) continue;
            } else {
                pos = parseInt(selectedPosition.value!);
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
                        position: pos,
                        video_source: ch.video_source,
                    });
                }
            }
        }
    }

    const baseThreshold = anyQueen ? 0.20 : 0.15;

    allEvents.sort((a, b) => a.time - b.time);

    const windowSize = 5;
    for (let i = 0; i < allEvents.length; i++) {
        const evt = allEvents[i];
        if (evt.ml_score !== undefined) {
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

    const MIN_HIGHLIGHT_GAP = 10;
    const result: PlayerHighlight[] = [];
    for (const setNum in eventsBySet) {
        eventsBySet[setNum].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
        const selected: PlayerHighlight[] = [];
        for (const evt of eventsBySet[setNum]) {
            const tooClose = selected.some(s => Math.abs(s.time - evt.time) < MIN_HIGHLIGHT_GAP);
            if (!tooClose) {
                selected.push(evt);
                if (selected.length >= targetPerSet) break;
            }
        }
        result.push(...selected);
    }

    const hasPositiveHighlight = result.some(h => {
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
        if (bestPositiveMove && !result.some(h => h.id === bestPositiveMove!.id)) {
            result.push(bestPositiveMove);
        }
    }

    result.sort((a, b) => a.time - b.time);
    return result;
}

/** Filter highlights to those matching the current video source. */
function filteredHighlights() {
    const cvs = currentVideoSource.value;
    const hl = playerHighlights.value;
    if (!cvs) return hl;
    return hl.filter(h => h.video_source === cvs);
}

export function nextHighlight(): void {
    const highlights = filteredHighlights();
    if (highlights.length === 0) return;

    const ct = getCurrentTime();
    let nextIndex;

    if (lastHighlightIndex.value >= 0 && lastHighlightIndex.value < highlights.length - 1) {
        const lastTime = highlights[lastHighlightIndex.value].time;
        if (Math.abs(ct - (lastTime - HIGHLIGHT_SEEK_BUFFER)) < 3) {
            nextIndex = lastHighlightIndex.value + 1;
        }
    }

    if (nextIndex === undefined) {
        for (let i = 0; i < highlights.length; i++) {
            if (highlights[i].time > ct + 0.5) {
                nextIndex = i;
                break;
            }
        }
    }

    if (nextIndex !== undefined && nextIndex < highlights.length) {
        lastHighlightIndex.value = nextIndex;
        seekTo(highlights[nextIndex].time - HIGHLIGHT_SEEK_BUFFER);
        const p = ytPlayer.value;
        if (p) p.playVideo();
    }
}

export function prevHighlight(): void {
    const highlights = filteredHighlights();
    if (highlights.length === 0) return;

    const ct = getCurrentTime();

    if (lastHighlightIndex.value >= 0) {
        const lastTime = highlights[lastHighlightIndex.value].time;

        if (ct > lastTime - 0.5) {
            seekTo(lastTime - HIGHLIGHT_SEEK_BUFFER);
            const p = ytPlayer.value;
            if (p) p.playVideo();
            return;
        }

        if (Math.abs(ct - (lastTime - HIGHLIGHT_SEEK_BUFFER)) < 2 && lastHighlightIndex.value > 0) {
            lastHighlightIndex.value--;
            seekTo(highlights[lastHighlightIndex.value].time - HIGHLIGHT_SEEK_BUFFER);
            const p = ytPlayer.value;
            if (p) p.playVideo();
            return;
        }
    }

    for (let i = highlights.length - 1; i >= 0; i--) {
        if (highlights[i].time < ct - 1) {
            lastHighlightIndex.value = i;
            seekTo(highlights[i].time - HIGHLIGHT_SEEK_BUFFER);
            const p = ytPlayer.value;
            if (p) p.playVideo();
            return;
        }
    }
}

export function checkHighlightAutoAdvance(
    toggleCallback: () => void,
): void {
    const highlights = filteredHighlights();
    if (!highlightModeEnabled.value || highlights.length === 0) return;
    if (lastHighlightIndex.value < 0) return;
    const ct = getCurrentTime();
    const currentHighlight = highlights[lastHighlightIndex.value];

    if (ct >= currentHighlight.time + HIGHLIGHT_PLAY_DURATION) {
        if (lastHighlightIndex.value < highlights.length - 1) {
            lastHighlightIndex.value++;
            seekTo(highlights[lastHighlightIndex.value].time - HIGHLIGHT_SEEK_BUFFER);
        } else {
            toggleCallback();
        }
    }
}
