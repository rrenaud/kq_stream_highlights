import type { Chapter, ModelTimelinePoint, TimelinePoint } from './types';

export function findClosestPoint(timeline: ModelTimelinePoint[], time: number): ModelTimelinePoint | null {
    if (!timeline || timeline.length === 0) return null;
    let lo = 0, hi = timeline.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (timeline[mid].t < time) lo = mid + 1;
        else hi = mid;
    }
    if (lo > 0 && Math.abs(timeline[lo - 1].t - time) < Math.abs(timeline[lo].t - time)) {
        lo--;
    }
    return timeline[lo];
}

export function findTimelinePoint(ch: Chapter, currentTime: number, field: string): ModelTimelinePoint | null {
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

export function buildTimelinePath(timeline: TimelinePoint[], startTime: number, duration: number, width: number, height: number, padding: number, flipForGold: boolean): string {
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
