import { POSITION_SVGS, POSITION_NAMES } from '../constants';

interface PositionIconProps {
    pos: string;
    size?: number;
}

export function PositionIcon({ pos, size = 18 }: PositionIconProps) {
    if (!POSITION_SVGS[pos]) return null;
    return (
        <img
            src={POSITION_SVGS[pos]}
            width={size}
            height={size}
            style="vertical-align: middle; margin-right: 4px;"
            alt={POSITION_NAMES[pos] || 'Position'}
        />
    );
}
