import { getCurrentTime, seekTo, chapters, queenKills, currentChapterIndex, lastQueenKillIndex, ytPlayer } from './state';
import type { Chapter } from './types';

export function findChapterAtTime(chapterList: Chapter[], time: number): number {
    for (let i = chapterList.length - 1; i >= 0; i--) {
        if (time >= chapterList[i].start_time) {
            return i;
        }
    }
    return -1;
}

export function jumpToChapter(index: number): void {
    const chs = chapters.value;
    if (index >= 0 && index < chs.length) {
        const targetTime = chs[index].start_time;
        console.log(`Jumping to chapter ${index}: ${chs[index].title} at ${targetTime}s`);
        seekTo(targetTime);
        currentChapterIndex.value = index;
        const p = ytPlayer.value;
        if (p) p.playVideo();
    }
}

export function prevChapter(): void {
    const idx = findChapterAtTime(chapters.value, getCurrentTime());
    if (idx > 0) {
        jumpToChapter(idx - 1);
    } else if (idx === 0) {
        jumpToChapter(0);
    }
}

export function nextChapter(): void {
    const chs = chapters.value;
    const idx = findChapterAtTime(chs, getCurrentTime());
    if (idx < chs.length - 1) {
        jumpToChapter(idx + 1);
    } else if (idx === -1 && chs.length > 0) {
        jumpToChapter(0);
    }
}

export function nextSet(): void {
    const chs = chapters.value;
    const idx = findChapterAtTime(chs, getCurrentTime());
    for (let i = idx + 1; i < chs.length; i++) {
        if (chs[i].is_set_start) {
            jumpToChapter(i);
            return;
        }
    }
}

export function prevSet(): void {
    const chs = chapters.value;
    const ct = getCurrentTime();
    const idx = findChapterAtTime(chs, ct);
    const currentSetStart = chs.findIndex((ch, i) =>
        i <= idx && ch.is_set_start &&
        (i === idx || !chs.slice(i + 1, idx + 1).some(c => c.is_set_start))
    );

    for (let i = idx - 1; i >= 0; i--) {
        if (chs[i].is_set_start) {
            if (i === currentSetStart && ct - chs[i].start_time < 2) {
                continue;
            }
            jumpToChapter(i);
            return;
        }
    }
    if (chs.length > 0) {
        jumpToChapter(0);
    }
}

export function findQueenKillIndexAtTime(time: number): number {
    const qks = queenKills.value;
    for (let i = qks.length - 1; i >= 0; i--) {
        if (qks[i].time <= time + 2) {
            return i;
        }
    }
    return -1;
}

export function nextQueenKill(): void {
    const qks = queenKills.value;
    if (qks.length === 0) return;

    const ct = getCurrentTime();
    let nextIndex;

    if (lastQueenKillIndex.value >= 0 && lastQueenKillIndex.value < qks.length - 1) {
        const lastKillTime = qks[lastQueenKillIndex.value].time;
        if (Math.abs(ct - (lastKillTime - 1)) < 3) {
            nextIndex = lastQueenKillIndex.value + 1;
        }
    }

    if (nextIndex === undefined) {
        nextIndex = findQueenKillIndexAtTime(ct) + 1;
    }

    if (nextIndex < qks.length) {
        lastQueenKillIndex.value = nextIndex;
        seekTo(qks[nextIndex].time - 1);
        const p = ytPlayer.value;
        if (p) p.playVideo();
    }
}

export function prevQueenKill(): void {
    const qks = queenKills.value;
    if (qks.length === 0) return;

    const ct = getCurrentTime();

    if (lastQueenKillIndex.value >= 0) {
        const lastKillTime = qks[lastQueenKillIndex.value].time;
        const targetTime = lastKillTime - 1;

        if (ct > lastKillTime - 0.5) {
            seekTo(targetTime);
            const p = ytPlayer.value;
            if (p) p.playVideo();
            return;
        }

        if (Math.abs(ct - targetTime) < 2 && lastQueenKillIndex.value > 0) {
            lastQueenKillIndex.value = lastQueenKillIndex.value - 1;
            seekTo(qks[lastQueenKillIndex.value].time - 1);
            const p = ytPlayer.value;
            if (p) p.playVideo();
            return;
        }
    }

    const idx = findQueenKillIndexAtTime(ct);
    if (idx >= 0) {
        lastQueenKillIndex.value = idx;
        seekTo(qks[idx].time - 1);
        const p = ytPlayer.value;
        if (p) p.playVideo();
    }
}
