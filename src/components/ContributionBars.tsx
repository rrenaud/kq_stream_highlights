import type { Chapter, ChapterData } from '../types';
import { CF_LABELS } from '../constants';
import { barGoesRight } from '../utils';
import { findTimelinePoint } from '../timeline';

interface ContributionBarsProps {
    ch: Chapter | null;
    currentTime: number;
    flipForGold: boolean;
    chapterData: ChapterData | null;
}

export function ContributionBars({ ch, currentTime, flipForGold, chapterData }: ContributionBarsProps) {
    if (!ch || !ch.model_timelines) return null;

    const point = findTimelinePoint(ch, currentTime, 'c');
    if (!point || !point.c) return null;

    const entries = [];
    for (const [key, delta] of Object.entries(point.c)) {
        const displayDelta = flipForGold ? -delta : delta;
        if (Math.abs(displayDelta) < 0.005) continue;
        const labelInfo = CF_LABELS[key] || [key, null];
        entries.push({ key, label: `${key} ${labelInfo[0]}`, team: labelInfo[1], delta: displayDelta, rawDelta: delta });
    }
    entries.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

    if (entries.length === 0) return null;

    const maxDelta = Math.max(0.10, ...entries.map(e => Math.abs(e.delta)));

    return (
        <div class="contribution-bars">
            <h4>What-if event values</h4>
            <div>
                {entries.map(e => {
                    const pct = (Math.abs(e.delta) / maxDelta) * 50;
                    const barClass = e.rawDelta > 0 ? 'positive' : 'negative';
                    const goesRight = barGoesRight(e.rawDelta, e.delta, ch.gold_on_left);
                    const barStyle = goesRight
                        ? `left: 50%; width: ${pct}%;`
                        : `left: ${50 - pct}%; width: ${pct}%;`;
                    const teamColor = e.team === 'blue' ? '#5ba3ec'
                        : e.team === 'gold' ? '#ffd700' : '#aaa';
                    const valueStr = (Math.abs(e.delta) * 100).toFixed(1) + '%';

                    return (
                        <div class="cf-bar-row" key={e.key}>
                            <span class="cf-bar-label" style={`color:${teamColor}`}>{e.label}</span>
                            <div class="cf-bar-track">
                                <div class="cf-bar-center"></div>
                                <div class={`cf-bar-fill ${barClass}`} style={barStyle}></div>
                            </div>
                            <span class="cf-bar-value">{valueStr}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
