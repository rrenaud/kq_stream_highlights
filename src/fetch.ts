/**
 * Fetch JSON data, trying .gz (gzip-compressed) first and falling back to plain JSON.
 * Uses the browser's DecompressionStream API for decompression.
 */
export async function fetchJSON<T = unknown>(url: string): Promise<T> {
    // URL is already .gz — decompress directly
    if (url.endsWith('.gz') && typeof DecompressionStream !== 'undefined') {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const ds = new DecompressionStream('gzip');
        const decompressed = resp.body!.pipeThrough(ds);
        const text = await new Response(decompressed).text();
        return JSON.parse(text);
    }

    // Try .gz version first (only if DecompressionStream is available)
    if (url.endsWith('.json') && typeof DecompressionStream !== 'undefined') {
        const resp = await fetch(url + '.gz');
        if (resp.ok) {
            const ds = new DecompressionStream('gzip');
            const decompressed = resp.body!.pipeThrough(ds);
            const text = await new Response(decompressed).text();
            return JSON.parse(text);
        }
        // .gz not found — fall through to plain JSON
    }

    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
}
