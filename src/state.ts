import { signal, computed } from '@preact/signals';
import type { Chapter, ChapterData, FlatQueenKill, PlayerHighlight, PositionId, UserInfo, VideoSource } from './types';
import { shouldFlipForGold } from './utils';

// Core player state
export const ytPlayer = signal<YT.Player | null>(null);
export const chapters = signal<Chapter[]>([]);
export const queenKills = signal<FlatQueenKill[]>([]);
export const currentChapterIndex = signal(-1);
export const lastQueenKillIndex = signal(-1);
export const currentTime = signal(0);

// Team & perspective
export const favoriteTeam = signal<'blue' | 'gold' | null>(null);

// Player/position selection
export const selectedPosition = signal<PositionId | null>(null);
export const selectedUserId = signal<string | null>(null);

// Highlights
export const playerHighlights = signal<PlayerHighlight[]>([]);
export const playerHighlightCount = signal(0);
export const playerLowlightCount = signal(0);
export const lastHighlightIndex = signal(-1);
export const highlightModeEnabled = signal(false);

// Data
export const users = signal<Record<string, UserInfo>>({});
export const videoId = signal<string | null>(null);
export const chapterData = signal<ChapterData | null>(null);
export const youtubeApiReady = signal(false);

// Multi-video support
export const videos = signal<Record<string, VideoSource>>({});
export const currentVideoSource = signal<string | null>(null);
export const cabFilter = signal<string | null>(null);

// Computed values
export const currentChapter = computed(() => {
    const idx = currentChapterIndex.value;
    const chs = chapters.value;
    return idx >= 0 && idx < chs.length ? chs[idx] : null;
});

export const flipForGold = computed(() => {
    const ch = currentChapter.value;
    if (!ch) return false;
    return shouldFlipForGold(ch, selectedPosition.value, selectedUserId.value, favoriteTeam.value);
});

// Player helpers (operate on ytPlayer signal directly)
export function getCurrentTime(): number {
    const p = ytPlayer.value;
    return p && p.getCurrentTime ? p.getCurrentTime() : 0;
}

export function getDuration(): number {
    const p = ytPlayer.value;
    return p && p.getDuration ? p.getDuration() : 0;
}

export function seekTo(seconds: number): void {
    const p = ytPlayer.value;
    if (p && p.seekTo) p.seekTo(seconds, true);
}

export function togglePlayPause(): void {
    const p = ytPlayer.value;
    if (!p) return;
    const playerState = p.getPlayerState();
    if (playerState === YT.PlayerState.PLAYING) {
        p.pauseVideo();
    } else {
        p.playVideo();
    }
}
