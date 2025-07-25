import { NextResponse } from 'next/server';
import axios from 'axios';

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000';

/*
 * POST /api/rfc/inference
 * Body options:
 * 1) { file: "test.csv" }               – run batch inference from CSV inside backend data/output/rfc/test
 * 2) { ...features }                      – run single inference with explicit features
 */
export async function POST(req: Request) {
  const url = new URL(req.url);
  const engine = url.searchParams.get("engine") ?? "python";
  try {
    const json = await req.json();

    // Case 1: file-based batch inference
    if (json.file) {
      const endpoint = engine === "c" ? "c" : "python";
      const { data } = await axios.post(
        `${BACKEND}/rfc/inference/${endpoint}`,
        {},
        { params: { file: json.file }, timeout: 60_000 }
      );
      return NextResponse.json(data);
    }

    // Case 2: single prediction
    const endpointSingle = engine === "c" ? "c" : "python";
    const { data } = await axios.post(
      `${BACKEND}/rfc/inference/${endpointSingle}`,
      json,
      { timeout: 60_000 }
    );
    return NextResponse.json(data);
  } catch (e: any) {
    return NextResponse.json({ success: false, error: e.message ?? 'Proxy error' }, { status: 500 });
  }
}
