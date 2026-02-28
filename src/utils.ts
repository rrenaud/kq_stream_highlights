import type { Chapter } from './types';

export function formatTime(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
        return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
}

export function isGoldTeam(positionId: string): boolean {
    const pos = parseInt(positionId);
    return pos % 2 === 1;
}

export function perspectiveDelta(delta: number, position: number | string | null): number {
    return (position && isGoldTeam(String(position))) ? -delta : delta;
}

export function probToColor(prob: number): string {
    const r = Math.round(249 + (59 - 249) * prob);
    const g = Math.round(115 + (130 - 115) * prob);
    const b = Math.round(22 + (246 - 22) * prob);
    return `rgba(${r},${g},${b},0.9)`;
}

export function getUserPositionInChapter(userId: string, chapter: Chapter): number | null {
    if (!chapter.users) return null;
    for (const [pos, uid] of Object.entries(chapter.users)) {
        if (String(uid) === String(userId)) {
            return parseInt(pos);
        }
    }
    return null;
}

export function getChapterPosition(ch: Chapter, selectedPosition: string | null, selectedUserId: string | null): string | null {
    let pos = selectedPosition;
    if (selectedUserId) {
        const p = getUserPositionInChapter(selectedUserId, ch);
        pos = p ? String(p) : null;
    }
    return pos;
}

export function shouldFlipForGold(ch: Chapter, selectedPosition: string | null, selectedUserId: string | null, favoriteTeam: 'blue' | 'gold' | null): boolean {
    if (favoriteTeam === 'blue') return false;
    if (favoriteTeam === 'gold') return true;
    const pos = getChapterPosition(ch, selectedPosition, selectedUserId);
    return !!pos && isGoldTeam(pos);
}

export function barGoesRight(rawDelta: number, displayDelta: number, goldOnLeft: boolean | undefined): boolean {
    return goldOnLeft !== undefined
        ? (rawDelta > 0) === !!goldOnLeft
        : displayDelta > 0;
}
