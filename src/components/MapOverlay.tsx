import type { Chapter, ChapterData, AffineTransform, MapStructureInfo } from '../types';
import { MAP_STRUCTURE, GATE_Y_OFFSET, OVERLAY_LINE_HEIGHT } from '../constants';
import { barGoesRight } from '../utils';
import { findTimelinePoint } from '../timeline';

interface MapOverlayProps {
    ch: Chapter | null;
    currentTime: number;
    flipForGold: boolean;
    chapterData: ChapterData | null;
}

function getOverlayPosition(key: string, mapInfo: MapStructureInfo, goldOnLeft: boolean | undefined, transform: AffineTransform | undefined, snailX: number | undefined, cfDict: Record<string, number> | undefined): [number, number][] | null {
    if (!mapInfo) return null;

    function toPercent(x: number, y: number, flipX: boolean): [number, number] {
        const px = flipX ? (1920 - x) : x;
        const py = 1080 - y;
        if (transform) {
            return [transform.a_x + transform.b_x * px, transform.a_y + transform.b_y * py];
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

    if (key === 'bb') return [toPercent(mapInfo.right_berries_centroid[0], mapInfo.right_berries_centroid[1], needsFlip)];
    if (key === 'gb') return [toPercent(mapInfo.left_berries_centroid[0], mapInfo.left_berries_centroid[1], needsFlip)];

    if (key === 'sb' || key === 'sg') {
        const sx = (snailX != null) ? snailX : mapInfo.snail_center[0];
        const sy = 1080 - mapInfo.snail_center[1];
        const yOff = key === 'sb' ? OVERLAY_LINE_HEIGHT : -OVERLAY_LINE_HEIGHT;
        return [toPercent(sx, sy + yOff, false)];
    }

    if (key === 'bqk' || key === 'gqk') {
        const gc = mapInfo.gold_eggs_centroid || [850, 899];
        const blueX = 960 + (960 - gc[0]);
        const ey = gc[1] - 45;
        if (key === 'gqk') return [toPercent(goldOnLeft ? gc[0] : blueX, ey, false)];
        return [toPercent(goldOnLeft ? blueX : gc[0], ey, false)];
    }

    if (key === 'bvwd' || key === 'bswd') {
        const h = mapInfo.blue_hive;
        return [toPercent(h[0], h[1] + (key === 'bswd' ? OVERLAY_LINE_HEIGHT : 0), needsFlip)];
    }
    if (key === 'gvwd' || key === 'gswd') {
        const h = mapInfo.gold_hive;
        return [toPercent(h[0], h[1] + (key === 'gswd' ? OVERLAY_LINE_HEIGHT : 0), needsFlip)];
    }

    if (key === 'bdw' || key === 'bsdw' || key === 'gdw' || key === 'gsdw') {
        const isBlue = key.startsWith('b');
        const yOff = isBlue ? -OVERLAY_LINE_HEIGHT : OVERLAY_LINE_HEIGHT;
        const positions: [number, number][] = [];
        mapInfo.maiden_info.forEach(([type, mx, my], idx) => {
            if (type !== 'maiden_wings') return;
            const flipKey = isBlue ? `mb${idx}` : `mg${idx}`;
            if (cfDict && !(flipKey in cfDict)) positions.push(toPercent(mx, my + yOff, needsFlip));
        });
        return positions.length > 0 ? positions : null;
    }

    if (key === 'bws' || key === 'gws') {
        const isBlue = key === 'bws';
        const positions: [number, number][] = [];
        mapInfo.maiden_info.forEach(([type, mx, my], idx) => {
            if (type !== 'maiden_speed') return;
            const flipKey = isBlue ? `mb${idx}` : `mg${idx}`;
            if (cfDict && !(flipKey in cfDict)) positions.push(toPercent(mx, my, needsFlip));
        });
        return positions.length > 0 ? positions : null;
    }

    return null;
}

export function MapOverlay({ ch, currentTime, flipForGold, chapterData }: MapOverlayProps) {
    if (!ch || !chapterData || !ch.model_timelines || !ch.map || ch.gold_on_left === undefined) {
        return <div class="cf-overlay"></div>;
    }

    const mapInfo = MAP_STRUCTURE[ch.map];
    if (!mapInfo) return <div class="cf-overlay"></div>;

    const point = findTimelinePoint(ch, currentTime, 'c');
    if (!point || !point.c) return <div class="cf-overlay"></div>;

    const positionedEntries: { key: string; delta: number; rawDelta: number; x: number; y: number }[] = [];
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

    if (positionedEntries.length === 0) return <div class="cf-overlay"></div>;

    const maxDelta = Math.max(0.10, ...positionedEntries.map(e => Math.abs(e.delta)));
    const BLUE_CF = 'rgba(59,130,246,0.9)';
    const ORANGE_CF = 'rgba(249,115,22,0.9)';

    return (
        <div class="cf-overlay">
            {positionedEntries.map((e, i) => {
                const fillPct = (Math.abs(e.delta) / maxDelta) * 50;
                const isBlueGood = e.rawDelta > 0;
                const color = isBlueGood ? BLUE_CF : ORANGE_CF;
                const goesRight = barGoesRight(e.rawDelta, e.delta, ch.gold_on_left);
                const barStyle = goesRight
                    ? `left:50%;width:${fillPct}%;background:${color};`
                    : `right:50%;width:${fillPct}%;background:${color};`;
                const pctText = `${e.key} ${(Math.abs(e.delta) * 100).toFixed(0)}%`;
                const labelColor = isBlueGood ? '#93c5fd' : '#fdba74';

                return (
                    <div key={i} class="cf-overlay-item" style={`left:${e.x}%;top:${e.y}%;`}>
                        <span class="cf-overlay-label" style={`color:${labelColor}`}>{pctText}</span>
                        <div class="cf-overlay-bar">
                            <div class="cf-overlay-bar-fill" style={barStyle}></div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
