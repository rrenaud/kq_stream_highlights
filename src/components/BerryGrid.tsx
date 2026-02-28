import type { Chapter } from '../types';
import { BERRY_DELTAS, MAX_FOOD } from '../constants';
import { findTimelinePoint } from '../timeline';
import { DiamondGrid } from './DiamondGrid';

interface BerryGridProps {
    ch: Chapter | null;
    currentTime: number;
    flipForGold: boolean;
}

export function BerryGrid({ ch, currentTime, flipForGold }: BerryGridProps) {
    if (!ch || !ch.model_timelines) return null;

    const point = findTimelinePoint(ch, currentTime, 'bg');
    if (!point || !point.bg) return null;

    const needsMirror = !!ch.gold_on_left;
    const bg = point.bg;
    const n = BERRY_DELTAS.length;

    const berryProbs: (number | null)[][] = [];
    for (let row = 0; row < n; row++) {
        berryProbs[row] = [];
        for (let col = 0; col < n; col++) {
            const idx = row * n + col;
            const raw = bg[idx];
            berryProbs[row][col] = (raw === null || raw === undefined) ? null : raw;
        }
    }

    const currentRow = 0, currentCol = 0;

    const leftTeam = ch.gold_on_left ? 'Gold' : 'Blue';
    const rightTeam = ch.gold_on_left ? 'Blue' : 'Gold';

    const bc = point.bc || [0, 0];
    const blueLabels = BERRY_DELTAS.map(d => String(Math.max(0, MAX_FOOD - bc[0] - d)));
    const goldLabels = BERRY_DELTAS.map(d => String(Math.max(0, MAX_FOOD - bc[1] - d)));
    const leftEdgeLabels = ch.gold_on_left ? goldLabels : blueLabels;
    const rightEdgeLabels = ch.gold_on_left ? blueLabels : goldLabels;

    return (
        <div class="egg-grid">
            <h4>Berry what-ifs</h4>
            <DiamondGrid
                probs={berryProbs} n={n} currentRow={currentRow} currentCol={currentCol}
                needsMirror={needsMirror} cellSize={40} fontSize={11}
                leftLabel={`${leftTeam} berries left`} rightLabel={`${rightTeam} berries left`}
                flipDisplay={flipForGold}
                leftEdgeLabels={leftEdgeLabels} rightEdgeLabels={rightEdgeLabels}
            />
        </div>
    );
}
