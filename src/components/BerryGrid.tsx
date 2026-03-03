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
    const bc = point.bc || [0, 0];

    // Per-team deltas: new format (bd) has ascending deltas with negative
    // historical values at low indices; old format uses fixed [0,1,2,3,4]
    const bd = point.bd;
    const blueDeltas = bd ? bd[0] : BERRY_DELTAS;
    const goldDeltas = bd ? bd[1] : BERRY_DELTAS;
    const n = blueDeltas.length;

    const berryProbs: (number | null)[][] = [];
    for (let row = 0; row < n; row++) {
        berryProbs[row] = [];
        for (let col = 0; col < n; col++) {
            const idx = row * n + col;
            const raw = bg[idx];
            berryProbs[row][col] = (raw === null || raw === undefined) ? null : raw;
        }
    }

    // Current position: where delta=0 is for each team
    const currentRow = blueDeltas.indexOf(0);
    const currentCol = goldDeltas.indexOf(0);

    const leftTeam = ch.gold_on_left ? 'Gold' : 'Blue';
    const rightTeam = ch.gold_on_left ? 'Blue' : 'Gold';

    const blueLabels = blueDeltas.map(d => String(Math.max(0, MAX_FOOD - bc[0] - d)));
    const goldLabels = goldDeltas.map(d => String(Math.max(0, MAX_FOOD - bc[1] - d)));
    const leftEdgeLabels = ch.gold_on_left ? goldLabels : blueLabels;
    const rightEdgeLabels = ch.gold_on_left ? blueLabels : goldLabels;

    // New format (bd): ascending deltas, historical (negative) at low indices → chasmBefore
    // Old format: ascending deltas [0..4], game-over at high indices → chasmAfter
    const chasmBeforeRow = bd ? currentRow : undefined;
    const chasmBeforeCol = bd ? currentCol : undefined;
    const chasmAfterRow = bd ? undefined : Math.min(n - 1, MAX_FOOD - bc[0] - 1);
    const chasmAfterCol = bd ? undefined : Math.min(n - 1, MAX_FOOD - bc[1] - 1);

    return (
        <div class="egg-grid">
            <h4>Berry what-ifs</h4>
            <DiamondGrid
                probs={berryProbs} n={n} currentRow={currentRow} currentCol={currentCol}
                needsMirror={needsMirror} cellSize={40} fontSize={11}
                leftLabel={`${leftTeam} berries left`} rightLabel={`${rightTeam} berries left`}
                flipDisplay={flipForGold}
                leftEdgeLabels={leftEdgeLabels} rightEdgeLabels={rightEdgeLabels}
                chasmBeforeRow={chasmBeforeRow} chasmBeforeCol={chasmBeforeCol}
                chasmAfterRow={chasmAfterRow} chasmAfterCol={chasmAfterCol}
                chasmGap={0.5}
            />
        </div>
    );
}
