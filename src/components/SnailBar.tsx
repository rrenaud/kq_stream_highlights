import type { Chapter, ChapterData } from '../types';
import { MAP_STRUCTURE, SNAIL_CHUNKS } from '../constants';
import { probToColor } from '../utils';
import { findTimelinePoint } from '../timeline';

interface SnailBarProps {
    ch: Chapter | null;
    currentTime: number;
    flipForGold: boolean;
    chapterData: ChapterData | null;
}

export function SnailBar({ ch, currentTime, flipForGold, chapterData }: SnailBarProps) {
    if (!ch || !chapterData || !ch.model_timelines || !ch.map || ch.gold_on_left === undefined) {
        return null;
    }

    const mapInfo = MAP_STRUCTURE[ch.map];
    if (!mapInfo) return null;

    const point = findTimelinePoint(ch, currentTime, 'sp');
    if (!point || !point.sp) return null;

    const sp = point.sp;
    const sc = point.sc;
    const st = point.st;

    const transform = chapterData.game_transform;
    const needsFlip = !ch.gold_on_left;

    // Compute left/right percentage positions for the bar
    // snail_left/snail_right are in game coordinates (gold_on_left reference frame)
    let leftPx = mapInfo.snail_left;
    let rightPx = mapInfo.snail_right;

    // When needsFlip (gold is on the right), mirror the x coordinates
    if (needsFlip) {
        leftPx = 1920 - mapInfo.snail_right;
        rightPx = 1920 - mapInfo.snail_left;
    }

    let leftPct: number, rightPct: number;
    if (transform) {
        leftPct = transform.a_x + transform.b_x * leftPx;
        rightPct = transform.a_x + transform.b_x * rightPx;
    } else {
        // No calibration — approximate using raw game coords over 1920px width
        leftPct = leftPx / 1920 * 100;
        rightPct = rightPx / 1920 * 100;
    }

    // Ensure left < right
    if (leftPct > rightPct) {
        [leftPct, rightPct] = [rightPct, leftPct];
    }

    // The sp array is in normalized order: index 0 = toward gold net (-0.5),
    // index 11 = toward blue net (+0.5). This is from gold_on_left perspective.
    // When flipForGold is true, we reverse so gold net is on left visually.
    // When flipForGold is false, blue perspective: index 0 (gold side) is left.
    // The normalized positions go from gold net to blue net.
    // On screen with gold_on_left: left = gold net, right = blue net (natural order).
    // When needsFlip (gold on right): left = blue net, right = gold net (reversed).
    // flipForGold reverses the display for gold's perspective.

    // Determine the display order of cells
    // sp[0] = gold net side, sp[11] = blue net side (normalized -0.5 to +0.5 * gold_sym)
    // In gold_on_left mode: screen left = gold net = sp[0], no reversal needed
    // In gold_on_right mode (!gold_on_left): screen left = blue net = sp[11], need reversal
    // flipForGold swaps perspective: if true, gold net should appear on left
    let displayProbs = [...sp];
    let displaySc = sc;

    // needsFlip means gold is on the right in the video
    // flipForGold means we want gold's perspective (gold on left)
    // If needsFlip XOR flipForGold, we need to reverse the array
    if (needsFlip !== flipForGold) {
        displayProbs.reverse();
        if (displaySc !== undefined) {
            displaySc = SNAIL_CHUNKS - 1 - displaySc;
        }
    }

    // For gold perspective, show 1-p (probability for gold instead of blue)
    if (flipForGold) {
        displayProbs = displayProbs.map(p => 1 - p);
    }

    const marginLeft = `${leftPct}%`;
    const marginRight = `${100 - rightPct}%`;

    // Compute takeover display probability
    let takeoverProb: number | undefined;
    if (st !== undefined) {
        takeoverProb = flipForGold ? 1 - st : st;
    }

    return (
        <div class="snail-bar" style={`margin-left:${marginLeft};margin-right:${marginRight};`}>
            <div class="snail-bar-row">
                {displayProbs.map((p, i) => {
                    const isCurrent = i === displaySc;
                    const pct = (p * 100).toFixed(0);
                    return (
                        <div
                            key={i}
                            class={`snail-bar-cell${isCurrent ? ' current' : ''}`}
                            style={`background:${probToColor(p)};`}
                        >
                            {pct}
                            {isCurrent && takeoverProb !== undefined && (
                                <span class="snail-takeover">{'\u21C5'}{(takeoverProb * 100).toFixed(0)}</span>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
