import { probToColor } from '../utils';

interface DiamondGridProps {
    probs: (number | null)[][];
    n: number;
    currentRow: number;
    currentCol: number;
    needsMirror: boolean;
    cellSize: number;
    fontSize: number;
    leftLabel: string;
    rightLabel: string;
    flipDisplay: boolean;
    leftEdgeLabels?: string[];
    rightEdgeLabels?: string[];
}

function contourBorderCSS(
    probs: (number | null)[][], row: number, col: number, n: number, needsMirror: boolean
): string {
    const p = probs[row][col];
    if (p === null) return '';
    const side = p >= 0.5;

    function crosses(r: number, c: number): boolean {
        if (r < 0 || r >= n || c < 0 || c >= n) return false;
        const np = probs[r][c];
        if (np === null) return false;
        return (np >= 0.5) !== side;
    }

    const borderStyle = '2px solid rgba(255,255,255,0.85)';
    const parts: string[] = [];

    if (!needsMirror) {
        if (crosses(row - 1, col)) parts.push(`border-top:${borderStyle};`);
        if (crosses(row, col + 1)) parts.push(`border-right:${borderStyle};`);
        if (crosses(row + 1, col)) parts.push(`border-bottom:${borderStyle};`);
        if (crosses(row, col - 1)) parts.push(`border-left:${borderStyle};`);
    } else {
        if (crosses(row, col - 1)) parts.push(`border-top:${borderStyle};`);
        if (crosses(row + 1, col)) parts.push(`border-right:${borderStyle};`);
        if (crosses(row, col + 1)) parts.push(`border-bottom:${borderStyle};`);
        if (crosses(row - 1, col)) parts.push(`border-left:${borderStyle};`);
    }

    return parts.join('');
}

export function DiamondGrid(props: DiamondGridProps) {
    const { probs, n, currentRow, currentCol, needsMirror, cellSize, fontSize,
            leftLabel, rightLabel, flipDisplay, leftEdgeLabels, rightEdgeLabels } = props;
    const step = cellSize * Math.SQRT2 / 2;
    const halfCell = cellSize / 2;
    const diagSpan = 2 * n - 1;
    const titleHeight = 16;
    const containerWidth = diagSpan * step + cellSize + 20;
    const containerHeight = titleHeight + diagSpan * step + cellSize;
    const cx = containerWidth / 2;
    const topPad = halfCell + titleHeight;

    const cells: preact.JSX.Element[] = [];
    for (let row = 0; row < n; row++) {
        for (let col = 0; col < n; col++) {
            const prob = probs[row][col];
            if (prob === null || prob === undefined) continue;

            let dx = col - row;
            if (needsMirror) dx = -dx;
            const dy = col + row;

            const x = cx + dx * step - halfCell;
            const y = topPad + dy * step - halfCell;

            const pct = flipDisplay ? Math.round((1 - prob) * 100) : Math.round(prob * 100);
            const bgColor = probToColor(prob);
            const isCurrent = (row === currentRow && col === currentCol);
            const currentClass = isCurrent ? ' egg-current-bold' : '';
            const contour = contourBorderCSS(probs, row, col, n, needsMirror);

            cells.push(
                <div
                    key={`${row}-${col}`}
                    class={`diamond-cell${currentClass}`}
                    style={`left:${x}px;top:${y}px;width:${cellSize + 1}px;height:${cellSize + 1}px;background:${bgColor};${contour}`}
                >
                    <span style={`font-size:${fontSize}px;`}>{pct}</span>
                </div>
            );
        }
    }

    const tickGap = 4;
    const tickFontSize = Math.max(9, fontSize - 2);
    const perpDist = (halfCell + tickGap) / Math.SQRT2;
    const edgeLabels: preact.JSX.Element[] = [];
    for (let i = 0; i < n; i++) {
        const cellCY = topPad + i * step;
        const leftText = leftEdgeLabels ? leftEdgeLabels[i] : String(i);
        const rightText = rightEdgeLabels ? rightEdgeLabels[i] : String(i);
        const lx = cx - i * step - perpDist;
        const ly = cellCY - perpDist;
        edgeLabels.push(
            <span key={`l-${i}`} class="diamond-axis-label"
                style={`right:${containerWidth - lx}px;top:${ly}px;transform:translateY(-50%);font-size:${tickFontSize}px;`}>
                {leftText}
            </span>
        );
        const rx = cx + i * step + perpDist;
        const ry = cellCY - perpDist;
        edgeLabels.push(
            <span key={`r-${i}`} class="diamond-axis-label"
                style={`left:${rx}px;top:${ry}px;transform:translateY(-50%);font-size:${tickFontSize}px;`}>
                {rightText}
            </span>
        );
    }

    return (
        <div class="diamond-grid-container" style={`width:${containerWidth}px;height:${containerHeight}px;`}>
            <span class="diamond-axis-label" style="left:0;top:0;">{leftLabel}</span>
            <span class="diamond-axis-label" style="right:0;top:0;">{rightLabel}</span>
            {cells}
            {edgeLabels}
        </div>
    );
}
