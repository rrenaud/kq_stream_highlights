/**
 * Fetch JSON data, trying .gz (gzip-compressed) first and falling back to plain JSON.
 * Uses the browser's DecompressionStream API for decompression.
 */

export interface FetchResult<T> {
    data: T;
    lastModified: string | null;
}

async function decompressResponse<T>(resp: Response): Promise<T> {
    // If the server set Content-Encoding: gzip, the browser already decompressed
    if (resp.headers.get('Content-Encoding')) {
        return resp.json();
    }
    const ds = new DecompressionStream('gzip');
    const decompressed = resp.body!.pipeThrough(ds);
    const text = await new Response(decompressed).text();
    return JSON.parse(text);
}

export async function fetchJSON<T = unknown>(url: string): Promise<FetchResult<T>> {
    // URL is already .gz — fetch and decompress if needed
    if (url.endsWith('.gz') && typeof DecompressionStream !== 'undefined') {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await decompressResponse<T>(resp);
        return { data, lastModified: resp.headers.get('Last-Modified') };
    }

    // Try .gz version first (only if DecompressionStream is available)
    if (url.endsWith('.json') && typeof DecompressionStream !== 'undefined') {
        const resp = await fetch(url + '.gz');
        if (resp.ok) {
            const data = await decompressResponse<T>(resp);
            return { data, lastModified: resp.headers.get('Last-Modified') };
        }
        // .gz not found — fall through to plain JSON
    }

    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data: T = await resp.json();
    return { data, lastModified: resp.headers.get('Last-Modified') };
}
