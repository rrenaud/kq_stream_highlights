import type { Chapter } from '../types';
import { findTimelinePoint } from '../timeline';
import { DiamondGrid } from './DiamondGrid';

interface EggGridProps {
    ch: Chapter | null;
    currentTime: number;
    flipForGold: boolean;
}

export function EggGrid({ ch, currentTime, flipForGold }: EggGridProps) {
    if (!ch || !ch.model_timelines) return null;

    const point = findTimelinePoint(ch, currentTime, 'eg');
    if (!point || !point.eg) return null;

    const needsMirror = !!ch.gold_on_left;
    const eg = point.eg;
    const ee = point.ee;

    const n = 3;
    const eggProbs: (number | null)[][] = [];
    let currentRow = -1, currentCol = -1;
    for (let row = 0; row < n; row++) {
        eggProbs[row] = [];
        for (let col = 0; col < n; col++) {
            const blueEggs = row;
            const goldEggs = col;
            const idx = blueEggs * n + goldEggs;
            eggProbs[row][col] = eg[idx];
            if (ee && blueEggs === ee[0] && goldEggs === ee[1]) {
                currentRow = row;
                currentCol = col;
            }
        }
    }

    const leftTeam = ch.gold_on_left ? 'Gold' : 'Blue';
    const rightTeam = ch.gold_on_left ? 'Blue' : 'Gold';

    return (
        <div class="egg-grid">
            <h4>Egg what-ifs</h4>
            <DiamondGrid
                probs={eggProbs} n={n} currentRow={currentRow} currentCol={currentCol}
                needsMirror={needsMirror} cellSize={66} fontSize={15}
                leftLabel={`${leftTeam} eggs`} rightLabel={`${rightTeam} eggs`}
                flipDisplay={flipForGold}
                chasmAfterRow={currentRow >= 0 ? currentRow : undefined}
                chasmAfterCol={currentCol >= 0 ? currentCol : undefined}
                chasmGap={0.5}
            />
        </div>
    );
}
