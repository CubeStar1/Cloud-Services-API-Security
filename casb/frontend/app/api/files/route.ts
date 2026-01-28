import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';
import path from 'path';

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000';

type BackendFile = {
  name: string;
  path: string;
  timestamp: number;
  content?: string;
};

type FrontendNode = {
  id: string;
  name: string;
  isSelectable?: boolean;
  children?: FrontendNode[];
};

// Convert flat list from backend to nested tree for frontend
function buildTreeFromFlatList(files: BackendFile[]): FrontendNode[] {
  if (files.length === 0) return [];

  // Infer relative paths by stripping everything before /data/
  const filesWithRelPath = files.map(f => {
    const normalized = f.path.split(path.sep).join('/');
    const dataIndex = normalized.lastIndexOf('/data/');
    let rel = normalized;
    if (dataIndex !== -1) {
      rel = normalized.substring(dataIndex + 6); // +6 for '/data/' length
    }
    return { ...f, relPath: rel };
  });
  
  const rootNodes: FrontendNode[] = [];
  
  // Helper to find or create directory node
  const getOrCreateDir = (pathParts: string[], currentLevel: FrontendNode[]) => {
    let current = currentLevel;
    let pathSoFar = '';
    
    for (let i = 0; i < pathParts.length - 1; i++) {
        const part = pathParts[i];
        pathSoFar = pathSoFar ? `${pathSoFar}/${part}` : part;
        
        let existing = current.find(n => n.name === part);
        if (!existing) {
            existing = {
                id: pathSoFar,
                name: part,
                isSelectable: true,
                children: [] 
            };
            current.push(existing);
        }
        if (!existing.children) existing.children = [];
        current = existing.children;
    }
    return current;
  };

  filesWithRelPath.forEach(f => {
      const parts = f.relPath.split('/');
      const fileName = parts[parts.length - 1];
      const parentChildren = getOrCreateDir(parts, rootNodes);
      
      parentChildren.push({
          id: f.relPath,
          name: fileName,
          isSelectable: !fileName.endsWith('.csv'),
      });
  });
  
  return rootNodes;
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const filePath = searchParams.get('file');
    
    // If specific file requested
    if (filePath) {
      const { data } = await axios.get(`${BACKEND}/files`, {
        params: { file: filePath }
      });
      
      if (data.error) {
         return NextResponse.json(data, { status: 404 });
      }
      
      const extension = data.name.split('.').pop();
       
      let responseData: any = {
          content: data.content,
          name: data.name,
      };
      
      if (extension === 'csv') {
          responseData.type = 'csv';
      } else {
          responseData.language = extension;
      }
      
      return NextResponse.json(responseData);
    }
    
    // Otherwise list files
    const { data } = await axios.get(`${BACKEND}/files`);
    const tree = buildTreeFromFlatList(data);
    
    return NextResponse.json({ tree });
    
  } catch (error: any) {
    console.error('Error in files proxy:', error.message);
    const status = error.response?.status ?? 500;
    return NextResponse.json({ 
        error: 'Failed to fetch files from backend' 
    }, { status });
  }
}
