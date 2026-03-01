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

    const midColor = probToColor(0.5);
    const blueColor = probToColor(1);
    const goldColor = probToColor(0);

    return (
        <div class="contribution-bars">
            <h4>What-if event values</h4>
            <div>
                {pairs.map(([pk, pair]) => {
                    const leftPct = pair.left ? (Math.abs(pair.left.delta) / maxDelta) * 100 : 0;
                    const rightPct = pair.right ? (Math.abs(pair.right.delta) / maxDelta) * 100 : 0;
                    const leftVal = pair.left ? (Math.abs(pair.left.delta) * 100).toFixed(1) : '';
                    const rightVal = pair.right ? (Math.abs(pair.right.delta) * 100).toFixed(1) : '';

                    const leftBarStyle = pair.left
                        ? `width:${leftPct}%;background:linear-gradient(to left,${midColor},${blueColor})`
                        : '';
                    const rightBarStyle = pair.right
                        ? `width:${rightPct}%;background:linear-gradient(to right,${midColor},${goldColor})`
                        : '';

                    const valueText = leftVal && rightVal
                        ? `${leftVal}\u2502${rightVal}`
                        : leftVal || rightVal;

                    return (
                        <div class="cf-bar-row" key={pk}>
                            <span class="cf-bar-label cf-bar-label-blue">{pair.left?.label || ''}</span>
                            <div class="cf-bar-half cf-bar-half-left">
                                {pair.left && <div class="cf-bar-fill" style={leftBarStyle}></div>}
                            </div>
                            <span class="cf-bar-value">{valueText}</span>
                            <div class="cf-bar-half">
                                {pair.right && <div class="cf-bar-fill" style={rightBarStyle}></div>}
                            </div>
                            <span class="cf-bar-label cf-bar-label-gold">{pair.right?.label || ''}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
