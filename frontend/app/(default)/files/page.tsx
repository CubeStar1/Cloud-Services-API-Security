'use client'

import { FileBrowser } from "@/components/files/file-browser"

export default function FilesPage() {
    return (
        <div className="flex-1 space-y-6 p-4 md:p-8 pt-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">Files</h2>
                    <p className="text-muted-foreground">
                        Browse and manage files in the data directory
                    </p>
                </div>
            </div>

            <FileBrowser />
        </div>
    )
}
