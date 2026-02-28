import { useEffect, useRef } from 'preact/hooks';
import { getCurrentTime, seekTo, togglePlayPause } from '../state';
import {
    prevChapter, nextChapter, prevSet, nextSet,
    nextQueenKill, prevQueenKill,
} from '../navigation';
import { nextHighlight, prevHighlight } from '../highlights';

interface KeyboardCallbacks {
    toggleHighlightMode: () => void;
    cycleTeamToggle: () => void;
    handlePositionSelect: (position: string) => void;
}

const KEY_TO_POSITION: Record<string, string> = {
    '1': '10', '2': '8', '3': '2', '4': '9', '5': '7',
    '6': '6', '7': '4', '8': '1', '9': '5', '0': '3',
};

export function useKeyboardShortcuts(callbacks: KeyboardCallbacks): void {
    const callbacksRef = useRef(callbacks);
    callbacksRef.current = callbacks;

    useEffect(() => {
        function handler(e: KeyboardEvent) {
            if ((e.target as HTMLElement).tagName === 'INPUT') return;
            const cb = callbacksRef.current;

            switch (e.key) {
                case ' ':
                    e.preventDefault();
                    togglePlayPause();
                    break;
                case 'ArrowLeft':
                    seekTo(getCurrentTime() - 5);
                    break;
                case 'ArrowRight':
                    seekTo(getCurrentTime() + 5);
                    break;
                case 'j':
                    seekTo(getCurrentTime() - 10);
                    break;
                case 'l':
                    seekTo(getCurrentTime() + 10);
                    break;
                case 'g':
                    nextChapter();
                    break;
                case 'G':
                    prevChapter();
                    break;
                case 's':
                    nextSet();
                    break;
                case 'S':
                    prevSet();
                    break;
                case 'e':
                    nextQueenKill();
                    break;
                case 'E':
                    prevQueenKill();
                    break;
                case 'h':
                    nextHighlight();
                    break;
                case 'H':
                    prevHighlight();
                    break;
                case 'a':
                case 'A':
                    cb.toggleHighlightMode();
                    break;
                case 't':
                case 'T':
                    cb.cycleTeamToggle();
                    break;
                case '1': case '2': case '3': case '4': case '5':
                case '6': case '7': case '8': case '9': case '0':
                    cb.handlePositionSelect(KEY_TO_POSITION[e.key]);
                    break;
            }
        }

        document.addEventListener('keydown', handler);
        return () => document.removeEventListener('keydown', handler);
    }, []);
}
