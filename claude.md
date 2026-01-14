# Claude Code Project Instructions

## Dev Log Protocol

Maintain `DEVLOG.md` with timestamped entries. Each entry gets a sequential ID.

When starting a new approach or significant code path:
- Create entry: `## [ID] - [timestamp] - [brief description]`
- Note the intent and approach

When abandoning/rewriting code:
- Add to the *original* entry: `⚠️ SUPERSEDED by [ID]` with brief reason
- The new entry should note: `Replaces [ID]`

Example:
```
## [003] - 2025-01-14 10:30 - Trying redis for session cache
Approach: Use redis pub/sub for...
⚠️ SUPERSEDED by [007] - Redis added too much latency for our use case

## [007] - 2025-01-14 14:20 - Switch to in-memory LRU cache
Replaces [003]. Using lru-cache package instead because...
```

After completing any significant code change, update DEVLOG.md before proceeding.
