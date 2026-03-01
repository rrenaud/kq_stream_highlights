import type { Chapter } from '../types';
import { getChapterPosition, shouldFlipForGold, probToColor } from '../utils';
import type { TimelinePoint } from '../types';
import { findHighImpactRanges } from '../highlights';

interface WinProbPlotProps {
    ch: Chapter;
    index: number;
    selectedPosition: string | null;
    selectedUserId: string | null;
    favoriteTeam: 'blue' | 'gold' | null;
    onSeek: (time: number, videoSource?: string) => void;
}

export function WinProbPlot({ ch, index, selectedPosition, selectedUserId, favoriteTeam, onSeek }: WinProbPlotProps) {
    const hasTimeline = ch.win_timeline && ch.win_timeline.length >= 2;
    const hasModels = ch.model_timelines && Object.keys(ch.model_timelines).length > 0;
    if (!hasTimeline && !hasModels) return null;

    const width = 280;
    const height = 36;
    const padding = 2;
    const startTime = ch.start_time;
    const endTime = ch.end_time;
    const duration = endTime - startTime;
    const chapterPosition = getChapterPosition(ch, selectedPosition, selectedUserId);
    const flipForGold = shouldFlipForGold(ch, selectedPosition, selectedUserId, favoriteTeam);
    const highImpactRanges = chapterPosition ? findHighImpactRanges(ch, chapterPosition) : [];

    const handleClick = (e: MouseEvent) => {
        e.stopPropagation();
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const clickX = (e.clientX - rect.left) / rect.width;
        const targetTime = ch.start_time + clickX * (ch.end_time - ch.start_time);
        onSeek(targetTime, ch.video_source);
    };

    const highlightRects = highImpactRanges.map((range, i) => {
        const x1 = padding + ((range.start - startTime) / duration) * (width - 2 * padding);
        const x2 = padding + ((range.end - startTime) / duration) * (width - 2 * padding);
        const rangeWidth = Math.max(x2 - x1, 6);
        const isGoodForPlayer = flipForGold ? range.delta < 0 : range.delta > 0;
        const color = isGoodForPlayer ? 'rgba(76, 175, 80, 0.4)' : 'rgba(244, 67, 54, 0.4)';
        return <rect key={i} x={x1 - 2} y="0" width={rangeWidth + 4} height={height} fill={color} />;
    });

    // Build colored line segments where each segment's color reflects the probability
    function buildColoredSegments(timeline: TimelinePoint[], keyPrefix: string): preact.JSX.Element[] {
        const segs: preact.JSX.Element[] = [];
        for (let i = 0; i < timeline.length - 1; i++) {
            const pt = timeline[i];
            const next = timeline[i + 1];
            const x1 = padding + ((pt.t - startTime) / duration) * (width - 2 * padding);
            const x2 = padding + ((next.t - startTime) / duration) * (width - 2 * padding);
            const p1 = flipForGold ? (1 - pt.p) : pt.p;
            const p2 = flipForGold ? (1 - next.p) : next.p;
            const y1 = height - padding - (p1 * (height - 2 * padding));
            const y2 = height - padding - (p2 * (height - 2 * padding));
            const avgP = (p1 + p2) / 2;
            segs.push(<line key={`${keyPrefix}-${i}`} x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={probToColor(avgP)} stroke-width="2" />);
        }
        return segs;
    }

    let segments: preact.JSX.Element[] = [];

    if (hasModels) {
        const modelNames = Object.keys(ch.model_timelines!);
        modelNames.forEach((name) => {
            const timeline = ch.model_timelines![name];
            if (!timeline || timeline.length < 2) return;
            segments = segments.concat(buildColoredSegments(timeline, name));
        });
    } else if (hasTimeline) {
        segments = buildColoredSegments(ch.win_timeline!, 'wt');
    }

    return (
        <div class="win-prob-plot" data-chapter={index} onClick={handleClick}>
            <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                {highlightRects}
                <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2}
                    stroke="#333" stroke-width="1" stroke-dasharray="2,2" />
                {segments}
            </svg>
        </div>
    );
}
