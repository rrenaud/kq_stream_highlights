import { ytPlayer, youtubeApiReady, videoId } from './state';

export interface YouTubeCallbacks {
    onReady: (event: YT.PlayerEvent) => void;
    onStateChange: (event: YT.OnStateChangeEvent) => void;
}

export function initializePlayer(callbacks: YouTubeCallbacks): void {
    if (!youtubeApiReady.value || !videoId.value || ytPlayer.value) return;

    ytPlayer.value = new YT.Player('player', {
        videoId: videoId.value,
        playerVars: {
            'autoplay': 0,
            'controls': 1,
            'rel': 0,
            'modestbranding': 1,
        },
        events: {
            'onReady': callbacks.onReady,
            'onStateChange': callbacks.onStateChange,
        }
    });
}

export function setupYouTubeAPI(callbacks: YouTubeCallbacks): void {
    function onYouTubeIframeAPIReady(): void {
        youtubeApiReady.value = true;
        initializePlayer(callbacks);
    }

    window.onYouTubeIframeAPIReady = onYouTubeIframeAPIReady;
    if (window.YT && window.YT.Player) {
        onYouTubeIframeAPIReady();
    }
}
