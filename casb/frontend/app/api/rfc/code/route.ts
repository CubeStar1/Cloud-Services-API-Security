import { NextResponse } from 'next/server';
import axios from 'axios';

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000';

// Where the code-gen files live inside the backend data folder
const CODEGEN_BASE = 'output/rfc/codegen';
const EM_BASE      = 'output/rfc/em-codegen';

/*
 * GET  /api/rfc/code?file=<name>
 *  – returns { name, path, timestamp, content }
 *
 * GET  /api/rfc/code        (no ?file)
 *  – returns { files: [...] } list for UI
 */
export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const fileName = searchParams.get('file');

  try {
    /* ---------- single-file view ---------- */
    if (fileName) {
      // Map UI name to relative path the backend understands
      let relPath = `${CODEGEN_BASE}/${fileName}`;
      if (fileName.startsWith('include/')) {
        relPath = `${EM_BASE}/${fileName}`;          // include/*.h
      }

      const { data } = await axios.get(`${BACKEND}/files`, {
        params: { file: relPath },
      });

      if (data.error) {
        return NextResponse.json({ error: data.error }, { status: 404 });
      }
      return NextResponse.json(data);                // backend already has name/path/timestamp/content
    }

    /* ---------- listing view ---------- */
    const resps = await Promise.all([
      axios.get(`${BACKEND}/files`, { params: { subdir: CODEGEN_BASE,         ext: 'c,txt' } }),
      axios.get(`${BACKEND}/files`, { params: { subdir: EM_BASE,              ext: 'c,txt' } }),
      axios.get(`${BACKEND}/files`, { params: { subdir: `${EM_BASE}/include`, ext: 'h'     } }),
    ]);

    // Merge & normalise filenames (prepend include/ for headers)
    const files = resps
      .flatMap(r => r.data as any[])
      .map((f: any) => ({
        ...f,
        name: f.path.includes('/include/') ? `include/${f.name}` : f.name,
      }))
      .sort((a: any, b: any) => a.name.localeCompare(b.name));

    return NextResponse.json({ files });
  } catch (e: any) {
    return NextResponse.json(
      { error: e.message ?? 'Proxy error' },
      { status: 500 },
    );
  }
}