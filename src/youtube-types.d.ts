// YouTube IFrame API type declarations
declare global {
    namespace YT {
        class Player {
            constructor(elementId: string, options: PlayerOptions);
            playVideo(): void;
            pauseVideo(): void;
            seekTo(seconds: number, allowSeekAhead: boolean): void;
            loadVideoById(videoId: string, startSeconds?: number): void;
            getCurrentTime(): number;
            getDuration(): number;
            getPlayerState(): number;
        }
        interface PlayerOptions {
            videoId?: string;
            playerVars?: Record<string, unknown>;
            events?: Record<string, (event: any) => void>;
        }
        interface PlayerEvent {
            target: Player;
        }
        interface OnStateChangeEvent {
            data: number;
            target: Player;
        }
        const PlayerState: {
            PLAYING: number;
            PAUSED: number;
            ENDED: number;
            BUFFERING: number;
        };
    }
    interface Window {
        onYouTubeIframeAPIReady?: () => void;
        YT?: typeof YT;
    }
}

export {};
