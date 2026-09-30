from pathlib import Path
import re, requests, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalRAG:
    """Small local TF-IDF retriever over the project's knowledge_base.md.

    It is intentionally transparent: each returned result contains the section
    title, similarity score and the exact local knowledge-base text.
    """
    def __init__(self, path="knowledge_base.md"):
        self.path = Path(path)
        text = self.path.read_text(encoding="utf-8")
        raw = [x.strip() for x in re.split(r"\n(?=## )", text) if x.strip()]
        self.chunks = []
        for chunk in raw:
            lines = chunk.splitlines()
            title = lines[0].lstrip('#').strip() if lines else "Untitled"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            self.chunks.append({"title": title, "text": body, "raw": chunk})
        corpus = [c["title"] + " " + c["text"] for c in self.chunks]
        self.v = TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)
        self.m = self.v.fit_transform(corpus)

    def retrieve_with_scores(self, q, k=3):
        q = (q or "").strip()
        if not q:
            return []
        scores = cosine_similarity(self.v.transform([q]), self.m)[0]
        order = np.argsort(scores)[::-1]
        # Return at most k, but don't silently claim irrelevant zero-score text is evidence.
        results = []
        for i in order[:k]:
            score = float(scores[i])
            results.append({
                "rank": len(results) + 1,
                "title": self.chunks[i]["title"],
                "score": score,
                "text": self.chunks[i]["text"],
            })
        return results

    def retrieve(self, q, k=3):
        return [r["raw"] for r in self._retrieve_raw(q, k)]

    def _retrieve_raw(self, q, k=3):
        q = (q or "").strip()
        if not q:
            return []
        scores = cosine_similarity(self.v.transform([q]), self.m)[0]
        order = np.argsort(scores)[::-1]
        return [self.chunks[i] for i in order[:k]]

def ollama(prompt, model="llama3.2"):
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": model, "stream": False,
        "messages": [
            {"role": "system", "content": "Give cautious educational explanations only. Use only the supplied evidence. Do not diagnose or prescribe. If evidence is insufficient, say so."},
            {"role": "user", "content": prompt}
        ]
    }, timeout=90)
    r.raise_for_status()
    return r.json()["message"]["content"]
