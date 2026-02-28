import type { Chapter, VideoSource } from '../types';
import { formatTime, isGoldTeam, getChapterPosition } from '../utils';
import { calculateKD, calculateNetWinProb } from '../highlights';
import { WinProbPlot } from './WinProbPlot';

interface ChapterListProps {
    chapters: Chapter[];
    currentChapterIndex: number;
    selectedPosition: string | null;
    selectedUserId: string | null;
    favoriteTeam: 'blue' | 'gold' | null;
    filter: string;
    videos?: Record<string, VideoSource>;
    cabFilter?: string | null;
    onJumpToChapter: (index: number) => void;
    onSeek: (time: number) => void;
}

export function ChapterList({ chapters, currentChapterIndex, selectedPosition, selectedUserId, favoriteTeam, filter, videos, cabFilter, onJumpToChapter, onSeek }: ChapterListProps) {
    const hasMultipleVideos = Object.keys(videos || {}).length > 1;
    const filterLower = filter.toLowerCase();

    return (
        <div class="chapter-list" id="chapterList">
            {chapters.map((ch, i) => {
                if (cabFilter && ch.video_source !== cabFilter) return null;
                const chapterPosition = getChapterPosition(ch, selectedPosition, selectedUserId);
                if (selectedUserId && !chapterPosition) return null;
                if (filter) {
                    const searchText = `${ch.map} ${ch.winner} ${ch.win_condition} ${ch.game_id}`.toLowerCase();
                    if (!searchText.includes(filterLower)) return null;
                }

                const winnerClass = ch.winner === 'gold' ? 'gold-win' : 'blue-win';
                const activeClass = i === currentChapterIndex ? 'active' : '';
                const setClass = ch.is_set_start ? 'set-start' : 'in-set';

                let stats: { kd: { kills: number; deaths: number }; playerNetProb: number | null } | null = null;
                if (chapterPosition) {
                    const kd = calculateKD(ch, chapterPosition);
                    const netProb = calculateNetWinProb(ch, chapterPosition);
                    if (kd) {
                        const playerNetProb = isGoldTeam(chapterPosition) ? -netProb! : netProb;
                        stats = { kd, playerNetProb };
                    }
                }

                return (
                    <div
                        key={ch.game_id}
                        class={`chapter-item ${winnerClass} ${activeClass} ${setClass}`}
                        data-index={i}
                        onClick={(e) => {
                            if ((e.target as HTMLElement).closest('.win-prob-plot')) return;
                            onJumpToChapter(i);
                        }}
                    >
                        {ch.is_set_start && ch.match_info && (
                            <div class="set-label">
                                <span class="blue">{ch.match_info.blue}</span> vs <span class="gold">{ch.match_info.gold}</span>
                            </div>
                        )}
                        <div class="chapter-title">
                            {ch.title}
                            {hasMultipleVideos && ch.video_source && videos?.[ch.video_source] && (
                                <span class="cab-badge">{videos[ch.video_source].label}</span>
                            )}
                        </div>
                        <div class="chapter-meta">
                            <span class={`winner ${ch.winner}`}>{ch.winner}</span> {ch.win_condition}
                            &nbsp;|&nbsp; {formatTime(ch.duration)}
                            {stats && (
                                <span>
                                    &nbsp;|&nbsp; <span class="kd-stats">{stats.kd.kills}/{stats.kd.deaths}</span>
                                    {stats.playerNetProb !== null && (
                                        <> <span class={stats.playerNetProb >= 0 ? 'good-prob' : 'bad-prob'}>
                                            {stats.playerNetProb >= 0 ? '+' : ''}{(stats.playerNetProb * 100).toFixed(0)}%
                                        </span></>
                                    )}
                                </span>
                            )}
                        </div>
                        <WinProbPlot
                            ch={ch}
                            index={i}
                            selectedPosition={selectedPosition}
                            selectedUserId={selectedUserId}
                            favoriteTeam={favoriteTeam}
                            onSeek={onSeek}
                        />
                    </div>
                );
            })}
        </div>
    );
}
