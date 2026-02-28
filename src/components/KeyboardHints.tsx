export function KeyboardHints() {
    return (
        <div class="keyboard-hints">
            <kbd>{'\u2190'}</kbd><kbd>{'\u2192'}</kbd> Seek 5s &nbsp;
            <kbd>J</kbd><kbd>L</kbd> Seek 10s &nbsp;
            <kbd>Space</kbd> Play/Pause<br />
            <kbd>G</kbd> Game (Shift: prev) &nbsp;
            <kbd>S</kbd> Set (Shift: prev) &nbsp;
            <kbd>E</kbd> Egg/queen kill (Shift: prev)<br />
            <kbd>1</kbd>-<kbd>0</kbd> Select position &nbsp;
            <kbd>H</kbd> Player highlight (Shift: prev) &nbsp;
            <kbd>A</kbd> Auto-play highlights &nbsp;
            <kbd>T</kbd> Team perspective
        </div>
    );
}
