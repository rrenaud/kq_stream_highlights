import type { PlayerHighlight, Chapter, UserInfo } from '../types';
import { POSITION_NAMES } from '../constants';
import { formatTime, perspectiveDelta, getUserPositionInChapter } from '../utils';
import { PositionIcon } from './PositionIcon';

interface HighlightDebugProps {
    highlights: PlayerHighlight[];
    highlightCount: number;
    lowlightCount: number;
    lastHighlightIndex: number;
    selectedUserId: string | null;
    selectedPosition: string | null;
    users: Record<string, UserInfo>;
    currentChapterIndex: number;
    chapters: Chapter[];
    onHighlightClick: (index: number) => void;
}

export function HighlightDebug({
    highlights, highlightCount, lowlightCount, lastHighlightIndex,
    selectedUserId, selectedPosition, users, currentChapterIndex, chapters,
    onHighlightClick,
}: HighlightDebugProps) {
    if (!selectedUserId && !selectedPosition) return <div class="highlight-debug" id="highlightDebug"></div>;

    if (highlights.length === 0) {
        return (
            <div class="highlight-debug" id="highlightDebug">
                <p class="highlight-debug-empty">No highlights found for this player</p>
            </div>
        );
    }

    let playerName = 'Selected Player';
    let playerIconPos: string | null = null;
    if (selectedUserId && users[selectedUserId]) {
        playerName = users[selectedUserId].name;
        if (currentChapterIndex >= 0) {
            const pos = getUserPositionInChapter(selectedUserId, chapters[currentChapterIndex]);
            if (pos) playerIconPos = String(pos);
        }
    } else if (selectedPosition) {
        playerName = POSITION_NAMES[selectedPosition] || `Position ${selectedPosition}`;
        playerIconPos = selectedPosition;
    }

    return (
        <div class="highlight-debug" id="highlightDebug">
            <h4>
                {playerIconPos && <PositionIcon pos={playerIconPos} size={20} />}
                {playerName} - <span class="good-prob">{highlightCount}</span> / <span class="bad-prob">{lowlightCount}</span>
            </h4>
            {highlights.map((h, idx) => {
                const displayDelta = perspectiveDelta(h.delta, h.position);
                const deltaClass = displayDelta >= 0 ? 'positive' : 'negative';
                const deltaStr = (displayDelta >= 0 ? '+' : '') + (displayDelta * 100).toFixed(0) + '%';
                const scoreStr = h.score ? `(${(h.score * 100).toFixed(0)})` : '';
                const valuesStr = h.values ? h.values.join(', ') : '';
                const eventIdStr = h.event_id ? `#${h.event_id}` : '';
                const isActive = idx === lastHighlightIndex;

                return (
                    <div key={idx}>
                        <div
                            class={`highlight-debug-item${isActive ? ' active' : ''}`}
                            data-highlight-index={idx}
                            onClick={() => onHighlightClick(idx)}
                        >
                            <span class="highlight-debug-pos">
                                {h.position && <PositionIcon pos={String(h.position)} size={14} />}
                            </span>
                            <span class="highlight-debug-time">{formatTime(h.time)}</span>
                            <span class={`highlight-debug-delta ${deltaClass}`}>{deltaStr}</span>
                            <span class="highlight-debug-type">{h.type || 'event'} {scoreStr}</span>
                            <span class="highlight-debug-game">Game {h.game_id} {eventIdStr}</span>
                        </div>
                        {valuesStr && <div class="highlight-debug-values">[{valuesStr}]</div>}
                    </div>
                );
            })}
        </div>
    );
}
