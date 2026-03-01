import type { Chapter, ChapterData } from '../types';
import { CF_LABELS } from '../constants';
import { probToColor } from '../utils';
import { findTimelinePoint } from '../timeline';

interface ContributionBarsProps {
    ch: Chapter | null;
    currentTime: number;
    flipForGold: boolean;
    chapterData: ChapterData | null;
}

function getPairKey(key: string): string {
    if (key.startsWith('mb') || key.startsWith('mg')) return 'gate' + key.slice(2);
    if (key === 'sb' || key === 'sg') return 'snail';
    if (key.startsWith('b')) return key.slice(1);
    if (key.startsWith('g')) return key.slice(1);
    return key;
}

interface Entry {
    key: string;
    label: string;
    team: string | null;
    delta: number;
    rawDelta: number;
}

export function ContributionBars({ ch, currentTime, flipForGold, chapterData }: ContributionBarsProps) {
    if (!ch || !ch.model_timelines) return null;

    const point = findTimelinePoint(ch, currentTime, 'c');
    if (!point || !point.c) return null;

    const entries: Entry[] = [];
    for (const [key, delta] of Object.entries(point.c)) {
        const displayDelta = flipForGold ? -delta : delta;
        if (Math.abs(displayDelta) < 0.005) continue;
        const labelInfo = CF_LABELS[key] || [key, null];
        entries.push({ key, label: labelInfo[0], team: labelInfo[1], delta: displayDelta, rawDelta: delta });
    }

    if (entries.length === 0) return null;

    const maxDelta = Math.max(0.10, ...entries.map(e => Math.abs(e.delta)));

    // Group entries into pairs, assigned by impact direction:
    // left = helps blue (rawDelta > 0), right = helps gold (rawDelta < 0)
    const pairMap = new Map<string, { left?: Entry, right?: Entry, maxAbs: number }>();
    for (const e of entries) {
        const pk = getPairKey(e.key);
        if (!pairMap.has(pk)) pairMap.set(pk, { maxAbs: 0 });
        const pair = pairMap.get(pk)!;
        if (e.rawDelta > 0) pair.left = e;   // helps blue → left
        else pair.right = e;                   // helps gold → right
        pair.maxAbs = Math.max(pair.maxAbs, Math.abs(e.delta));
    }

    const pairs = [...pairMap.entries()].sort((a, b) => b[1].maxAbs - a[1].maxAbs);

    // Gradient based on absolute probability: spine = current prob, extending toward blue (left) or gold (right)
    const currentP = point.p;
    const spineColor = probToColor(currentP);
    const leftOuterColor = probToColor(Math.min(1, currentP + maxDelta));
    const rightOuterColor = probToColor(Math.max(0, currentP - maxDelta));

    // 50% marker: appears in the left or right half-track depending on which side of 50% the current prob is
    const delta50 = Math.abs(currentP - 0.5);
    const show50 = delta50 > 0.001 && delta50 <= maxDelta;
    const marker50Side: 'left' | 'right' = currentP < 0.5 ? 'left' : 'right';
    const marker50Pct = (delta50 / maxDelta) * 100;
    const markerStyle = `position:absolute;top:0;bottom:0;width:1px;background:white;opacity:0.6;${
        marker50Side === 'left' ? `right:${marker50Pct}%` : `left:${marker50Pct}%`
    }`;

    return (
        <div class="contribution-bars">
            <h4>What-if event values</h4>
            <div>
                {pairs.map(([pk, pair]) => {
                    const leftPct = pair.left ? (Math.abs(pair.left.delta) / maxDelta) * 100 : 0;
                    const rightPct = pair.right ? (Math.abs(pair.right.delta) / maxDelta) * 100 : 0;
                    const leftVal = pair.left ? (Math.abs(pair.left.delta) * 100).toFixed(1) : '';
                    const rightVal = pair.right ? (Math.abs(pair.right.delta) * 100).toFixed(1) : '';

                    // If the bar is X% of the track, size the gradient at (100/X * 100)% so
                    // equal probability offsets map to equal pixel positions across all rows.
                    const bgSizeLeft = leftPct > 0 ? `${10000 / leftPct}%` : '100%';
                    const bgSizeRight = rightPct > 0 ? `${10000 / rightPct}%` : '100%';

                    const leftBarStyle = pair.left
                        ? `width:${leftPct}%;background-image:linear-gradient(to left,${spineColor},${leftOuterColor});background-size:${bgSizeLeft} 100%;background-position:right`
                        : '';
                    const rightBarStyle = pair.right
                        ? `width:${rightPct}%;background-image:linear-gradient(to right,${spineColor},${rightOuterColor});background-size:${bgSizeRight} 100%;background-position:left`
                        : '';

                    const valueText = leftVal && rightVal
                        ? `${leftVal}\u2502${rightVal}`
                        : leftVal || rightVal;

                    return (
                        <div class="cf-bar-row" key={pk}>
                            <span class="cf-bar-label cf-bar-label-blue">{pair.left?.label || ''}</span>
                            <div class="cf-bar-half cf-bar-half-left">
                                {pair.left && <div class="cf-bar-fill" style={leftBarStyle}></div>}
                                {show50 && marker50Side === 'left' && <div style={markerStyle}></div>}
                            </div>
                            <span class="cf-bar-value">{valueText}</span>
                            <div class="cf-bar-half">
                                {pair.right && <div class="cf-bar-fill" style={rightBarStyle}></div>}
                                {show50 && marker50Side === 'right' && <div style={markerStyle}></div>}
                            </div>
                            <span class="cf-bar-label cf-bar-label-gold">{pair.right?.label || ''}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
