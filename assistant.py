import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import openai
import PyPDF2

CHUNK_SIZE = 2800
TOP_K = 3
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"  # Ajuste conforme a disponibilidade da sua conta


@dataclass
class TextChunk:
    text: str
    embedding: List[float]


class HemogramaAssistant:
    def __init__(self, pdf_path: str, openai_api_key: str) -> None:
        self.pdf_path = pdf_path
        openai.api_key = openai_api_key
        self.chunks = self._load_document_chunks(pdf_path)

    def _load_document_chunks(self, pdf_path: str) -> List[TextChunk]:
        raw_text = self._extract_text_from_pdf(pdf_path)
        fragments = self._split_text(raw_text, CHUNK_SIZE)
        return [TextChunk(text=fragment, embedding=self._embed_text(fragment)) for fragment in fragments]

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise RuntimeError(
                f"Não foi possível abrir o PDF '{pdf_path}'. Verifique o arquivo e tente novamente. "
                f"Detalhes: {exc}"
            )
        return "\n\n".join(text_parts).strip()

    def _split_text(self, text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                breakpoint = text.rfind("\n", start, end)
                if breakpoint <= start:
                    breakpoint = end
                end = breakpoint
            chunks.append(text[start:end].strip())
            start = end
        return [chunk for chunk in chunks if chunk]

    def _embed_text(self, text: str) -> List[float]:
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _select_context(self, question: str) -> str:
        question_embedding = self._embed_text(question)
        scored: List[Tuple[float, str]] = []
        for chunk in self.chunks:
            score = self._cosine_similarity(question_embedding, chunk.embedding)
            scored.append((score, chunk.text))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [text for _, text in scored[:TOP_K]]
        return "\n\n---\n\n".join(selected)

    def answer_question(self, user_input: str) -> str:
        context = self._select_context(user_input)
        system_prompt = (
            "Você é um assistente de saúde que interpreta hemogramas usando apenas o conteúdo da documentação fornecida. "
            "Seja gentil, claro e não invente respostas. "
            "Se não houver informação suficiente no documento, diga isso de forma amigável."
        )

        user_prompt = (
            "Documento de referência:\n"
            f"{context}\n\n"
            "Pergunta do usuário:\n"
            f"{user_input}\n\n"
            "Responda com base no documento, explique se houver sinais de risco de diabetes ou outras doenças, "
            "e use linguagem clara para o paciente."
        )

        try:
            completion = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=650,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            return (
                "Desculpe, ocorreu um problema ao gerar a resposta com o serviço de IA. "
                "Verifique sua conexão e a chave de API, ou tente novamente mais tarde. "
                f"Detalhes técnicos: {exc}"
            )
