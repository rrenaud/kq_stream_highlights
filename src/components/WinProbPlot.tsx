import type { Chapter } from '../types';
import { MODEL_COLORS } from '../constants';
import { getChapterPosition, shouldFlipForGold } from '../utils';
import { buildTimelinePath } from '../timeline';
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

    let paths: preact.JSX.Element[] = [];
    let legendItems: preact.JSX.Element[] = [];

    if (hasModels) {
        const modelNames = Object.keys(ch.model_timelines!);
        if (hasTimeline) {
            const basePathD = buildTimelinePath(ch.win_timeline!, startTime, duration, width, height, padding, flipForGold);
            paths.push(<path key="baseline" d={basePathD} fill="none" stroke="#888" stroke-width="1" stroke-dasharray="3,2" opacity="0.6" />);
        }
        modelNames.forEach((name, idx) => {
            const timeline = ch.model_timelines![name];
            if (!timeline || timeline.length < 2) return;
            const color = MODEL_COLORS[idx % MODEL_COLORS.length];
            const pathD = buildTimelinePath(timeline, startTime, duration, width, height, padding, flipForGold);
            paths.push(<path key={name} d={pathD} fill="none" stroke={color} stroke-width="1.5" />);
        });
        legendItems = modelNames.map((name, idx) => {
            const color = MODEL_COLORS[idx % MODEL_COLORS.length];
            return <span key={name} style={`color:${color}; margin-right:8px; font-size:10px;`}>{'\u25CF'} {name}</span>;
        });
        if (hasTimeline) {
            legendItems.push(<span key="hivemind" style="color:#888; font-size:10px;">{'\u2504'} HiveMind</span>);
        }
    } else {
        const pathD = buildTimelinePath(ch.win_timeline!, startTime, duration, width, height, padding, flipForGold);
        paths.push(<path key="single" d={pathD} fill="none" stroke="#888" stroke-width="1.5" />);
    }

    return (
        <div class="win-prob-plot" data-chapter={index} onClick={handleClick}>
            <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                {highlightRects}
                <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2}
                    stroke="#333" stroke-width="1" stroke-dasharray="2,2" />
                {paths}
            </svg>
            {legendItems.length > 0 && (
                <div style="display:flex; flex-wrap:wrap; gap:2px; margin-top:2px;">
                    {legendItems}
                </div>
            )}
        </div>
    );
}
