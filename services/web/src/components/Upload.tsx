import React, { useState } from "react"

export default function Upload({ apiBase }: { apiBase: string }) {
  const [files, setFiles] = useState<FileList | null>(null)
  const [log, setLog] = useState<string>("")

  const onUpload = async () => {
    if (!files || files.length === 0) return
    const form = new FormData()
    Array.from(files).forEach(f => form.append("files", f))
    setLog("Uploading...")
    const res = await fetch(`${apiBase}/ingest/upload`, { method: "POST", body: form })
    const json = await res.json()
    setLog(JSON.stringify(json, null, 2))
  }

  return (
    <div className="border rounded p-4 space-y-3 bg-white">
      <h2 className="font-semibold">Upload PDFs</h2>
      <input type="file" accept="application/pdf" multiple onChange={(e) => setFiles(e.target.files)} />
      <button className="px-3 py-1 rounded bg-black text-white" onClick={onUpload}>Ingest</button>
      <pre className="text-xs bg-gray-100 p-2 rounded overflow-auto max-h-64">{log}</pre>
    </div>
  )
}
