import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

/* GET ?type=models|predictions
 * models        -> data/output/rfc/models  (joblib files)
 * (default)     -> data/output/codebert/predictions (csv files)
 */
export async function GET(req: NextRequest) {
  try {
    const backend = process.env.BACKEND_URL ?? 'http://localhost:8000';
    const type = req.nextUrl.searchParams.get('type');
    let params: any;

    switch (type) {
      case 'models': params = { subdir: 'output/rfc/models', ext: 'joblib' }
        break;
      case 'predictions': params = { subdir: 'output/codebert/predictions', ext: 'csv' }
        break;
      case 'test': params = { subdir: 'output/rfc/test', ext: 'csv' }
        break;
      default:
        params = { subdir: 'output/codebert/predictions', ext: 'csv' }
        break;
    }

    const { data } = await axios.get(`${backend}/files`, { params });
    return NextResponse.json({ files: data });
  } catch (err: any) {
    console.error('files-list error', err.message);
    return NextResponse.json({ error: 'Failed to fetch files' }, { status: 500 });
  }
}

/* POST { file } → triggers RFC Python training */
export async function POST(req: NextRequest) {
  try {
    const { file } = await req.json();
    if (!file)
      return NextResponse.json({ error: 'No file specified' }, { status: 400 });

    const backend = process.env.BACKEND_URL ?? 'http://localhost:8000';
    const { data } = await axios.post(
      `${backend}/rfc/train/python`,
      { input_file: file },
      { timeout: 60 * 60 * 1000 } // 1 hour
    );
    return NextResponse.json(data);
  } catch (err: any) {
    const msg = err.response?.data?.error ?? err.message ?? 'Training failed';
    const output = err.response?.data?.output ?? [];
    return NextResponse.json({ error: msg, output }, { status: 500 });
  }
}