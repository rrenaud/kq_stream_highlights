import type { MapStructureInfo } from './types';

export const POSITION_NAMES: Record<string, string> = {
    '1': 'Gold Queen', '2': 'Blue Queen',
    '3': 'Gold Stripes', '4': 'Gold Skull', '5': 'Gold Abs', '6': 'Gold Checkers',
    '7': 'Blue Stripes', '8': 'Blue Skull', '9': 'Blue Abs', '10': 'Blue Checkers'
};

export const POSITION_ICONS: Record<string, string> = {
    '1': '\u{1F451}',   // Gold Queen
    '2': '\u{1F451}',   // Blue Queen
    '3': '\u2630',    // Gold Stripes (trigram)
    '4': '\u{1F480}',   // Gold Skull
    '5': '\u{1F4AA}',   // Gold Abs
    '6': '\u25A6',    // Gold Checkers
    '7': '\u2630',    // Blue Stripes
    '8': '\u{1F480}',   // Blue Skull
    '9': '\u{1F4AA}',   // Blue Abs
    '10': '\u25A6'    // Blue Checkers
};

export const GOLD_COLOR = '#ffc107';
export const BLUE_COLOR = '#2196f3';
export const DARK_BG = '#1a1a1a';

const SVG_TEMPLATES: Record<string, (color: string) => string> = {
    crown: (color: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><path d="M5 16L3 5l5.5 5L12 4l3.5 6L21 5l-2 11H5zm14 3c0 .6-.4 1-1 1H6c-.6 0-1-.4-1-1v-1h14v1z"/></svg>`)}`,
    stripes: (color: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><rect x="3" y="4" width="18" height="3" rx="1"/><rect x="3" y="10" width="18" height="3" rx="1"/><rect x="3" y="16" width="18" height="3" rx="1"/></svg>`)}`,
    skull: (color: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><path d="M12 2C6.5 2 2 6.5 2 12v3.5c0 1.4 1.1 2.5 2.5 2.5H6v-3h2v4h3v-4h2v4h3v-4h2v3h1.5c1.4 0 2.5-1.1 2.5-2.5V12c0-5.5-4.5-10-10-10zm-3 12a2 2 0 110-4 2 2 0 010 4zm6 0a2 2 0 110-4 2 2 0 010 4z"/></svg>`)}`,
    abs: (color: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}"><ellipse cx="12" cy="5" rx="4" ry="3"/><path d="M8 9h8v13H8V9z"/><rect x="9" y="10" width="2.5" height="3" fill="${DARK_BG}"/><rect x="12.5" y="10" width="2.5" height="3" fill="${DARK_BG}"/><rect x="9" y="14" width="2.5" height="3" fill="${DARK_BG}"/><rect x="12.5" y="14" width="2.5" height="3" fill="${DARK_BG}"/><rect x="9" y="18" width="2.5" height="2.5" fill="${DARK_BG}"/><rect x="12.5" y="18" width="2.5" height="2.5" fill="${DARK_BG}"/></svg>`)}`,
    checkers: (color: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect width="24" height="24" fill="${color}"/><rect x="0" y="0" width="6" height="6" fill="${DARK_BG}"/><rect x="12" y="0" width="6" height="6" fill="${DARK_BG}"/><rect x="6" y="6" width="6" height="6" fill="${DARK_BG}"/><rect x="18" y="6" width="6" height="6" fill="${DARK_BG}"/><rect x="0" y="12" width="6" height="6" fill="${DARK_BG}"/><rect x="12" y="12" width="6" height="6" fill="${DARK_BG}"/><rect x="6" y="18" width="6" height="6" fill="${DARK_BG}"/><rect x="18" y="18" width="6" height="6" fill="${DARK_BG}"/></svg>`)}`
};

export const POSITION_SVGS: Record<string, string> = {
    '1': SVG_TEMPLATES.crown(GOLD_COLOR),
    '2': SVG_TEMPLATES.crown(BLUE_COLOR),
    '3': SVG_TEMPLATES.stripes(GOLD_COLOR),
    '4': SVG_TEMPLATES.skull(GOLD_COLOR),
    '5': SVG_TEMPLATES.abs(GOLD_COLOR),
    '6': SVG_TEMPLATES.checkers(GOLD_COLOR),
    '7': SVG_TEMPLATES.stripes(BLUE_COLOR),
    '8': SVG_TEMPLATES.skull(BLUE_COLOR),
    '9': SVG_TEMPLATES.abs(BLUE_COLOR),
    '10': SVG_TEMPLATES.checkers(BLUE_COLOR)
};

export const CF_LABELS: Record<string, [string, string | null]> = {
    'bqk': ['Queen Kill', 'blue'],
    'gqk': ['Queen Kill', 'gold'],
    'bb': ['Berry', 'blue'],
    'gb': ['Berry', 'gold'],
    'bswd': ['Speed Warrior Dies', 'blue'],
    'bvwd': ['Warrior Dies', 'blue'],
    'gswd': ['Speed Warrior Dies', 'gold'],
    'gvwd': ['Warrior Dies', 'gold'],
    'bsdw': ['Speed Gets Wings', 'blue'],
    'bdw': ['Gets Wings', 'blue'],
    'bws': ['Gets Speed', 'blue'],
    'gsdw': ['Speed Gets Wings', 'gold'],
    'gdw': ['Gets Wings', 'gold'],
    'gws': ['Gets Speed', 'gold'],
    'sb': ['Snail \u2192 Blue', null],
    'sg': ['Snail \u2192 Gold', null],
};
// Add per-maiden labels dynamically
for (let i = 0; i < 5; i++) {
    CF_LABELS[`mb${i}`] = [`Gate ${i} \u2192 Blue`, 'blue'];
    CF_LABELS[`mg${i}`] = [`Gate ${i} \u2192 Gold`, 'gold'];
}

export const MAP_STRUCTURE: Record<string, MapStructureInfo> = {
    'Day': {
        maiden_info: [
            ['maiden_speed', 410, 860],
            ['maiden_speed', 1510, 860],
            ['maiden_wings', 560, 260],
            ['maiden_wings', 960, 500],
            ['maiden_wings', 1360, 260]
        ],
        left_berries_centroid: [830, 937],
        right_berries_centroid: [1090, 937],
        snail_center: [960, 1010],
        blue_hive: [1860, 980],
        gold_hive: [60, 980],
        gold_eggs_centroid: [850, 899]
    },
    'Dusk': {
        maiden_info: [
            ['maiden_speed', 340, 140],
            ['maiden_speed', 1580, 140],
            ['maiden_wings', 310, 620],
            ['maiden_wings', 960, 140],
            ['maiden_wings', 1610, 620]
        ],
        left_berries_centroid: [800, 685],
        right_berries_centroid: [1120, 685],
        snail_center: [960, 870],
        blue_hive: [1860, 980],
        gold_hive: [60, 980],
        gold_eggs_centroid: [746, 532]
    },
    'Night': {
        maiden_info: [
            ['maiden_speed', 170, 740],
            ['maiden_speed', 1750, 740],
            ['maiden_wings', 700, 260],
            ['maiden_wings', 960, 700],
            ['maiden_wings', 1220, 260]
        ],
        left_berries_centroid: [170, 96],
        right_berries_centroid: [1750, 96],
        snail_center: [960, 970],
        blue_hive: [1860, 980],
        gold_hive: [60, 980],
        gold_eggs_centroid: [97, 55]
    },
    'Twilight': {
        maiden_info: [
            ['maiden_speed', 410, 860],
            ['maiden_speed', 1510, 860],
            ['maiden_wings', 550, 260],
            ['maiden_wings', 960, 820],
            ['maiden_wings', 1370, 260]
        ],
        left_berries_centroid: [158, 322],
        right_berries_centroid: [1762, 322],
        snail_center: [960, 1010],
        blue_hive: [1860, 980],
        gold_hive: [60, 980],
        gold_eggs_centroid: [164, 52]
    }
};

export const MODEL_COLORS = ['#e94560', '#5ba3ec', '#50c878', '#f5a623'];

export const HIGHLIGHT_SEEK_BUFFER = 4.5;
export const GATE_Y_OFFSET = 4;
export const OVERLAY_LINE_HEIGHT = 20;
export const HIGHLIGHT_PLAY_DURATION = 6.0;
export const BERRY_DELTAS = [0, 1, 2, 3, 4];
export const MAX_FOOD = 12;
