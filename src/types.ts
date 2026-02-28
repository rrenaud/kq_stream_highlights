export type MapName = 'Day' | 'Dusk' | 'Night' | 'Twilight';
export type Team = 'gold' | 'blue';
export type PositionId = string;

export interface AffineTransform {
    a_x: number;
    b_x: number;
    a_y: number;
    b_y: number;
}

export interface QueenKill {
    time: number;
    victim: number;
}

export interface PlayerEvent {
    time: number;
    delta: number;
    type: string;
    positions?: number[];
    id?: number;
    values?: string[];
    ml_score?: number;
}

export interface KillEvent {
    killer: number;
    victim: number;
    time: number;
}

export interface TimelinePoint {
    t: number;
    p: number;
}

export interface ModelTimelinePoint extends TimelinePoint {
    c?: Record<string, number>;
    eg?: number[];
    ee?: [number, number];
    bg?: (number | null)[];
    bc?: [number, number];
    sx?: number;
    sp?: number[];    // 12 probabilities at each discrete snail position
    sc?: number;      // current snail position index (0-11)
    st?: number;      // takeover probability (opponent takes snail)
}

export interface VideoSource {
    video_id: string;
    label: string;
}

export interface MatchInfo {
    blue: string;
    gold: string;
}

export interface Chapter {
    title: string;
    start_time: number;
    end_time: number;
    duration: number;
    map: MapName;
    winner: Team;
    win_condition: string;
    game_id: number;
    set_number: number;
    is_set_start: boolean;
    match_info?: MatchInfo;
    queen_kills?: QueenKill[];
    player_events?: PlayerEvent[];
    kill_events?: KillEvent[];
    win_timeline?: TimelinePoint[];
    model_timelines?: Record<string, ModelTimelinePoint[]>;
    users?: Record<string, string | number>;
    gold_on_left?: boolean;
    hivemind_url: string;
    video_source?: string;
    _timelineFields?: Record<string, boolean>;
}

export interface UserInfo {
    name: string;
    scene?: string;
}

export interface ChapterData {
    video_id: string | null;
    chapters: Chapter[];
    users: Record<string, UserInfo>;
    game_transform?: AffineTransform;
    videos?: Record<string, VideoSource>;
}

export interface FlatQueenKill {
    time: number;
    victim: number;
    game_id: number;
    video_source?: string;
}

export interface PlayerHighlight {
    time: number;
    delta: number;
    type: string;
    game_id: number;
    set_number: number;
    event_id?: number;
    values?: string[];
    position: number | null;
    ml_score?: number;
    score?: number;
    id?: number;
    video_source?: string;
}

export interface HighImpactRange {
    start: number;
    end: number;
    delta: number;
}

export interface KDStats {
    kills: number;
    deaths: number;
}

export interface MapStructureInfo {
    maiden_info: [string, number, number][];
    left_berries_centroid: [number, number];
    right_berries_centroid: [number, number];
    snail_center: [number, number];
    blue_hive: [number, number];
    gold_hive: [number, number];
    gold_eggs_centroid: [number, number];
}

export interface SpeedGates {
    left: { gx: number; gy: number };
    right: { gx: number; gy: number };
}

export interface CalibrationClick {
    x: number;
    y: number;
}

export interface OverlayEntry {
    key: string;
    delta: number;
    rawDelta: number;
    x: number;
    y: number;
}

export interface DiamondGridOptions {
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
