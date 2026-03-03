import { probToColor } from '../utils';

interface DiamondGridProps {
    probs: (number | null)[][];
    nRows: number;
    nCols: number;
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
    chasmBeforeRow?: number;
    chasmBeforeCol?: number;
    chasmAfterRow?: number;
    chasmAfterCol?: number;
    chasmGap?: number;
}

function contourBorderCSS(
    probs: (number | null)[][], row: number, col: number,
    nRows: number, nCols: number, needsMirror: boolean
): string {
    const p = probs[row][col];
    if (p === null) return '';
    const side = p >= 0.5;

    function crosses(r: number, c: number): boolean {
        if (r < 0 || r >= nRows || c < 0 || c >= nCols) return false;
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
    const { probs, nRows, nCols, currentRow, currentCol, needsMirror, cellSize, fontSize,
            leftLabel, rightLabel, flipDisplay, leftEdgeLabels, rightEdgeLabels,
            chasmBeforeRow, chasmBeforeCol, chasmAfterRow, chasmAfterCol,
            chasmGap = 0 } = props;
    const step = cellSize * Math.SQRT2 / 2;
    const halfCell = cellSize / 2;
    const rowGapBefore = (chasmBeforeRow !== undefined && chasmBeforeRow > 0) ? chasmGap : 0;
    const colGapBefore = (chasmBeforeCol !== undefined && chasmBeforeCol > 0) ? chasmGap : 0;
    const rowGapAfter = (chasmAfterRow !== undefined && chasmAfterRow < nRows - 1) ? chasmGap : 0;
    const colGapAfter = (chasmAfterCol !== undefined && chasmAfterCol < nCols - 1) ? chasmGap : 0;
    const diagSpan = nRows + nCols - 1 + rowGapBefore + colGapBefore + rowGapAfter + colGapAfter;
    const titleHeight = 16;
    const containerWidth = diagSpan * step + cellSize + 20;
    const containerHeight = titleHeight + diagSpan * step + cellSize;
    const cx = containerWidth / 2;
    const topShift = (rowGapBefore + colGapBefore) * step;
    const topPad = halfCell + titleHeight + topShift;

    const cells: preact.JSX.Element[] = [];
    for (let row = 0; row < nRows; row++) {
        for (let col = 0; col < nCols; col++) {
            const prob = probs[row][col];
            const isNull = prob === null || prob === undefined;

            let effRow = row;
            if (chasmBeforeRow !== undefined && row < chasmBeforeRow) effRow -= chasmGap;
            if (chasmAfterRow !== undefined && row > chasmAfterRow) effRow += chasmGap;
            let effCol = col;
            if (chasmBeforeCol !== undefined && col < chasmBeforeCol) effCol -= chasmGap;
            if (chasmAfterCol !== undefined && col > chasmAfterCol) effCol += chasmGap;

            let dx = effCol - effRow;
            if (needsMirror) dx = -dx;
            const dy = effCol + effRow;

            const x = cx + dx * step - halfCell;
            const y = topPad + dy * step - halfCell;

            const beyondChasm =
                (chasmBeforeRow !== undefined && row < chasmBeforeRow) ||
                (chasmBeforeCol !== undefined && col < chasmBeforeCol) ||
                (chasmAfterRow !== undefined && row > chasmAfterRow) ||
                (chasmAfterCol !== undefined && col > chasmAfterCol);

            if (isNull) {
                cells.push(
                    <div
                        key={`${row}-${col}`}
                        class="diamond-cell"
                        style={`left:${x}px;top:${y}px;width:${cellSize + 1}px;height:${cellSize + 1}px;background:#333;opacity:0.4;`}
                    />
                );
            } else {
                const pct = flipDisplay ? Math.round((1 - prob) * 100) : Math.round(prob * 100);
                const bgColor = probToColor(prob);
                const isCurrent = (row === currentRow && col === currentCol);
                const currentClass = isCurrent ? ' egg-current-bold' : '';
                // Note: contour borders may cross chasms — this is a known cosmetic
                // imperfection that's acceptable since chasms are already visually distinct.
                const contour = contourBorderCSS(probs, row, col, nRows, nCols, needsMirror);
                const opacity = beyondChasm ? 'opacity:0.5;' : '';

                cells.push(
                    <div
                        key={`${row}-${col}`}
                        class={`diamond-cell${currentClass}`}
                        style={`left:${x}px;top:${y}px;width:${cellSize + 1}px;height:${cellSize + 1}px;background:${bgColor};${contour}${opacity}`}
                    >
                        <span style={`font-size:${fontSize}px;`}>{pct}</span>
                    </div>
                );
            }
        }
    }

    const tickGap = 4;
    const tickFontSize = Math.max(9, fontSize - 2);
    const perpDist = (halfCell + tickGap) / Math.SQRT2;
    const nLeft = needsMirror ? nCols : nRows;
    const nRight = needsMirror ? nRows : nCols;
    const leftChasmBefore = needsMirror ? chasmBeforeCol : chasmBeforeRow;
    const leftChasmAfter = needsMirror ? chasmAfterCol : chasmAfterRow;
    const rightChasmBefore = needsMirror ? chasmBeforeRow : chasmBeforeCol;
    const rightChasmAfter = needsMirror ? chasmAfterRow : chasmAfterCol;
    const edgeLabels: preact.JSX.Element[] = [];
    for (let i = 0; i < nLeft; i++) {
        let leftEffI = i;
        if (leftChasmBefore !== undefined && i < leftChasmBefore) leftEffI -= chasmGap;
        if (leftChasmAfter !== undefined && i > leftChasmAfter) leftEffI += chasmGap;
        const leftText = leftEdgeLabels ? leftEdgeLabels[i] : String(i);
        const lx = cx - leftEffI * step - perpDist;
        const ly = topPad + leftEffI * step - perpDist;
        edgeLabels.push(
            <span key={`l-${i}`} class="diamond-axis-label"
                style={`right:${containerWidth - lx}px;top:${ly}px;transform:translateY(-50%);font-size:${tickFontSize}px;`}>
                {leftText}
            </span>
        );
    }
    for (let i = 0; i < nRight; i++) {
        let rightEffI = i;
        if (rightChasmBefore !== undefined && i < rightChasmBefore) rightEffI -= chasmGap;
        if (rightChasmAfter !== undefined && i > rightChasmAfter) rightEffI += chasmGap;
        const rightText = rightEdgeLabels ? rightEdgeLabels[i] : String(i);
        const rx = cx + rightEffI * step + perpDist;
        const ry = topPad + rightEffI * step - perpDist;
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
