import CitationPreview from "./CitationPreview";

export default function ResultCard({ apiBase, hit, query }: { apiBase: string; hit: any; query: string }) {
    return (
        <div className="p-4 rounded-2xl shadow border">
        <div className="text-xs opacity-70 mb-1">
        {hit.citation?.title || hit.citation?.url || "Citation"}
        </div>

        <div className="mb-3 text-sm whitespace-pre-wrap">{hit.text}</div>

        {hit.citation?.id && (
            <CitationPreview apiBase={apiBase} citation={hit.citation} query={query} className="mt-2" />
        )}
        </div>
    );
}
