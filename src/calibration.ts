import { chapterData, ytPlayer } from './state';
import type { CalibrationClick, Chapter, SpeedGates } from './types';
import { MAP_STRUCTURE } from './constants';

export interface CalibrationState {
    clicks: CalibrationClick[];
    gates: SpeedGates | null;
}

export function createCalibrationState(): CalibrationState {
    return {
        clicks: [],
        gates: null,
    };
}

export function getSpeedGates(ch: Chapter | null): SpeedGates | null {
    if (!ch || !ch.map) return null;
    const mapInfo = MAP_STRUCTURE[ch.map];
    if (!mapInfo) return null;

    const speedGates = mapInfo.maiden_info
        .filter(m => m[0] === 'maiden_speed')
        .map(m => ({ gx: m[1], gy: m[2] }));
    if (speedGates.length < 2) return null;

    const needsFlip = !ch.gold_on_left;
    const effectiveGates = speedGates.map(g => ({
        gx: needsFlip ? (1920 - g.gx) : g.gx,
        gy: g.gy
    }));

    effectiveGates.sort((a, b) => a.gx - b.gx);
    return { left: effectiveGates[0], right: effectiveGates[effectiveGates.length - 1] };
}

export function handleCalibrationClick(
    calState: CalibrationState,
    xPct: number,
    yPct: number,
    btn: HTMLElement,
    overlay: HTMLElement,
    updateOverlayCallback: (time: number) => void,
): void {
    calState.clicks.push({ x: xPct, y: yPct });

    function crosshairHTML(x: number, y: number): string {
        return `<div style="position:absolute;left:${x}%;top:0;width:2px;height:100%;background:rgba(255,255,0,0.7);pointer-events:none;"></div>
            <div style="position:absolute;top:${y}%;left:0;width:100%;height:2px;background:rgba(255,255,0,0.7);pointer-events:none;"></div>
            <div style="position:absolute;left:${x}%;top:${y}%;width:16px;height:16px;border-radius:50%;background:yellow;border:2px solid red;transform:translate(-50%,-50%);pointer-events:none;z-index:999;"></div>`;
    }

    if (calState.clicks.length === 1) {
        overlay.innerHTML = crosshairHTML(xPct, yPct);
        btn.textContent = 'Click on RIGHT speed gate';
    } else if (calState.clicks.length === 2) {
        const ox1 = calState.clicks[0].x, oy1 = calState.clicks[0].y;
        const ox2 = calState.clicks[1].x, oy2 = calState.clicks[1].y;
        const gx1 = calState.gates!.left.gx, gy1 = calState.gates!.left.gy;
        const gx2 = calState.gates!.right.gx, gy2 = calState.gates!.right.gy;

        const b_x = (ox2 - ox1) / (gx2 - gx1);
        const a_x = ox1 - b_x * gx1;
        const b_y = b_x * 16 / 9;
        const a_y = oy1 - b_y * (1080 - gy1);

        const gameTransform = {
            a_x: Math.round(a_x * 1000) / 1000,
            b_x: Math.round(b_x * 100000) / 100000,
            a_y: Math.round(a_y * 1000) / 1000,
            b_y: Math.round(b_y * 100000) / 100000
        };

        const cd = chapterData.value;
        if (cd) {
            chapterData.value = { ...cd, game_transform: gameTransform };
        }
        console.log('game_transform:', JSON.stringify(gameTransform));
        const json = JSON.stringify(gameTransform);
        navigator.clipboard.writeText(`"game_transform": ${json}`).then(() => {
            btn.textContent = 'Calibrated! (copied to clipboard)';
        }).catch(() => {
            btn.textContent = 'Calibrated! Check console';
        });
        btn.style.background = '#4caf50';
        calState.clicks = [];
        calState.gates = null;
        overlay.style.pointerEvents = 'none';
        overlay.style.cursor = '';
        overlay.style.background = '';
        const p = ytPlayer.value;
        if (p && p.getCurrentTime) {
            updateOverlayCallback(p.getCurrentTime());
        }
    }
}
