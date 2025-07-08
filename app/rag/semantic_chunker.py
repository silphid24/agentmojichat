"""
의미 기반 청킹 시스템 (Semantic Chunking)
문장 임베딩과 유사도를 기반으로 더 자연스러운 청킹을 수행
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import nltk
from sentence_transformers import SentenceTransformer

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.logging import logger


class ChunkingStrategy(Enum):
    """청킹 전략 종류"""

    FIXED_SIZE = "fixed_size"  # 고정 크기 청킹
    SEMANTIC = "semantic"  # 의미 기반 청킹
    STRUCTURAL = "structural"  # 구조 기반 청킹
    ADAPTIVE = "adaptive"  # 적응형 청킹


@dataclass
class ChunkMetadata:
    """청크 메타데이터"""

    chunk_id: str
    source_file: str
    chunk_index: int
    total_chunks: int
    chunk_type: str
    semantic_score: float
    start_position: int
    end_position: int
    section_title: str = ""  # None 대신 빈 문자열
    structure_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """None 값을 제거하고 유효한 타입만 포함하는 딕셔너리 반환"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, (str, int, float, bool)):
                    result[key] = value
                else:
                    result[key] = str(value)
        return result


class SemanticChunker:
    """의미 기반 청킹 시스템"""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_chunk_size: int = 200,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.5,
        overlap_size: int = 50,
        use_structure_hints: bool = True,
    ):
        self.embedding_model_name = embedding_model
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.overlap_size = overlap_size
        self.use_structure_hints = use_structure_hints

        # 모델 초기화
        self.sentence_model = None
        self._initialize_models()

        # 구조 패턴 정의
        self.structure_patterns = {
            "title": re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE),
            "section": re.compile(r"^##\s+(.+)$", re.MULTILINE),
            "subsection": re.compile(r"^###\s+(.+)$", re.MULTILINE),
            "bullet": re.compile(r"^\s*[-*+]\s+(.+)$", re.MULTILINE),
            "number": re.compile(r"^\s*\d+\.\s+(.+)$", re.MULTILINE),
        }

    def _initialize_models(self):
        """모델 초기화"""
        try:
            # NLTK 데이터 다운로드 (필요시)
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            # 문장 임베딩 모델 로드
            self.sentence_model = SentenceTransformer(self.embedding_model_name)
            logger.info(
                f"Semantic chunker initialized with model: {self.embedding_model_name}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize semantic chunker: {e}")
            raise

    def chunk_document(
        self,
        text: str,
        source_file: str = "unknown",
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    ) -> List[Document]:
        """문서를 청킹하여 Document 객체 리스트 반환"""

        if strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunking(text, source_file)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(text, source_file)
        elif strategy == ChunkingStrategy.STRUCTURAL:
            return self._structural_chunking(text, source_file)
        elif strategy == ChunkingStrategy.ADAPTIVE:
            return self._adaptive_chunking(text, source_file)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _fixed_size_chunking(self, text: str, source_file: str) -> List[Document]:
        """고정 크기 청킹 (기존 방식)"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_text(text)
        documents = []

        for i, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                chunk_id=f"{source_file}_{i}",
                source_file=source_file,
                chunk_index=i,
                total_chunks=len(chunks),
                chunk_type="fixed",
                semantic_score=0.0,
                start_position=0,
                end_position=len(chunk),
            )

            doc = Document(page_content=chunk, metadata=metadata.to_dict())
            documents.append(doc)

        return documents

    def _semantic_chunking(self, text: str, source_file: str) -> List[Document]:
        """의미 기반 청킹"""
        try:
            # 문장 분할
            sentences = self._split_sentences(text)
            if len(sentences) < 2:
                return self._fixed_size_chunking(text, source_file)

            # 문장 임베딩 생성
            embeddings = self.sentence_model.encode(sentences)

            # 의미적 유사도 기반 청킹
            chunks = self._group_sentences_by_similarity(sentences, embeddings)

            # Document 객체 생성
            documents = []
            for i, chunk_sentences in enumerate(chunks):
                chunk_text = " ".join(chunk_sentences)

                # 청크가 너무 길면 분할
                if len(chunk_text) > self.max_chunk_size:
                    sub_chunks = self._split_long_chunk(chunk_text)
                    for j, sub_chunk in enumerate(sub_chunks):
                        metadata = ChunkMetadata(
                            chunk_id=f"{source_file}_{i}_{j}",
                            source_file=source_file,
                            chunk_index=len(documents),
                            total_chunks=0,  # 나중에 업데이트
                            chunk_type="semantic_sub",
                            semantic_score=self._calculate_chunk_coherence(sub_chunk),
                            start_position=text.find(sub_chunk),
                            end_position=text.find(sub_chunk) + len(sub_chunk),
                        )

                        doc = Document(
                            page_content=sub_chunk, metadata=metadata.to_dict()
                        )
                        documents.append(doc)
                else:
                    metadata = ChunkMetadata(
                        chunk_id=f"{source_file}_{i}",
                        source_file=source_file,
                        chunk_index=i,
                        total_chunks=0,  # 나중에 업데이트
                        chunk_type="semantic",
                        semantic_score=self._calculate_chunk_coherence(chunk_text),
                        start_position=text.find(chunk_text),
                        end_position=text.find(chunk_text) + len(chunk_text),
                    )

                    doc = Document(page_content=chunk_text, metadata=metadata.to_dict())
                    documents.append(doc)

            # total_chunks 업데이트
            for doc in documents:
                doc.metadata["total_chunks"] = len(documents)

            logger.info(
                f"Semantic chunking created {len(documents)} chunks for {source_file}"
            )
            return documents

        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}, falling back to fixed size")
            return self._fixed_size_chunking(text, source_file)

    def _structural_chunking(self, text: str, source_file: str) -> List[Document]:
        """구조 기반 청킹 (제목, 섹션 등)"""
        try:
            # 구조 요소 식별
            structure_points = self._identify_structure(text)

            if not structure_points:
                return self._semantic_chunking(text, source_file)

            # 구조를 기반으로 분할
            chunks = self._split_by_structure(text, structure_points)

            documents = []
            for i, (chunk_text, section_info) in enumerate(chunks):
                chunk_text = chunk_text.strip()
                if len(chunk_text) < self.min_chunk_size:
                    # 작은 청크는 다음 청크와 합치거나 최소 크기로 유지
                    if i + 1 < len(chunks):
                        # 다음 청크와 합치기
                        next_chunk_text, next_section_info = chunks[i + 1]
                        combined_text = chunk_text + "\n\n" + next_chunk_text.strip()
                        chunks[i + 1] = (combined_text, section_info)
                        continue
                    elif len(chunk_text) < 50:  # 너무 작으면 건너뛰기
                        continue

                metadata = ChunkMetadata(
                    chunk_id=f"{source_file}_{i}",
                    source_file=source_file,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    chunk_type="structural",
                    semantic_score=self._calculate_chunk_coherence(chunk_text),
                    start_position=section_info.get("start", 0),
                    end_position=section_info.get("end", len(chunk_text)),
                    section_title=section_info.get("title", ""),
                    structure_level=section_info.get("level", 0),
                )

                doc = Document(page_content=chunk_text, metadata=metadata.to_dict())
                documents.append(doc)

            logger.info(
                f"Structural chunking created {len(documents)} chunks for {source_file}"
            )

            # 0개 청크 방지: 최소 1개는 반환
            if not documents:
                logger.warning("Structural chunking created 0 chunks, using fallback")
                return self._fixed_size_chunking(text, source_file)

            return documents

        except Exception as e:
            logger.warning(f"Structural chunking failed: {e}, falling back to semantic")
            return self._semantic_chunking(text, source_file)

    def _adaptive_chunking(self, text: str, source_file: str) -> List[Document]:
        """적응형 청킹 (문서 특성에 따라 전략 선택)"""
        # 문서 특성 분석
        doc_analysis = self._analyze_document(text)

        # 최적 전략 선택
        if doc_analysis["has_clear_structure"]:
            return self._structural_chunking(text, source_file)
        elif doc_analysis["is_conversational"]:
            return self._semantic_chunking(text, source_file)
        else:
            return self._fixed_size_chunking(text, source_file)

    def _split_sentences(self, text: str) -> List[str]:
        """문장 분할"""
        try:
            sentences = nltk.sent_tokenize(text)
            # 너무 짧은 문장 제거 및 정리
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return sentences
        except Exception as e:
            logger.warning(f"NLTK sentence splitting failed: {e}, using simple split")
            # 폴백: 간단한 문장 분할
            sentences = re.split(r"[.!?]+\s+", text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _group_sentences_by_similarity(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[List[str]]:
        """유사도 기반 문장 그룹핑"""
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])

        for i in range(1, len(sentences)):
            # 이전 문장과의 유사도 계산
            prev_emb = embeddings[i - 1]
            curr_emb = embeddings[i]
            similarity = np.dot(prev_emb, curr_emb) / (
                np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb)
            )

            sentence_len = len(sentences[i])

            # 청킹 결정 로직
            if (
                similarity >= self.similarity_threshold
                and current_size + sentence_len <= self.max_chunk_size
            ):
                # 현재 청크에 추가
                current_chunk.append(sentences[i])
                current_size += sentence_len
            else:
                # 새로운 청크 시작
                if current_size >= self.min_chunk_size:
                    chunks.append(current_chunk)
                else:
                    # 너무 작은 청크는 다음 청크와 합침
                    if chunks:
                        chunks[-1].extend(current_chunk)
                    else:
                        chunks.append(current_chunk)

                current_chunk = [sentences[i]]
                current_size = sentence_len

        # 마지막 청크 추가
        if current_chunk:
            if current_size >= self.min_chunk_size or not chunks:
                chunks.append(current_chunk)
            else:
                chunks[-1].extend(current_chunk)

        return chunks

    def _split_long_chunk(self, text: str) -> List[str]:
        """긴 청크를 분할"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=len,
        )
        return splitter.split_text(text)

    def _calculate_chunk_coherence(self, text: str) -> float:
        """청크의 의미적 일관성 점수 계산"""
        try:
            sentences = self._split_sentences(text)
            if len(sentences) < 2:
                return 1.0

            embeddings = self.sentence_model.encode(sentences)
            similarities = []

            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(sim)

            return float(np.mean(similarities))

        except Exception as e:
            logger.warning(f"Failed to calculate coherence: {e}")
            return 0.5

    def _identify_structure(self, text: str) -> List[Dict[str, Any]]:
        """문서 구조 요소 식별"""
        structure_points = []

        for structure_type, pattern in self.structure_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                structure_points.append(
                    {
                        "type": structure_type,
                        "text": match.group(1) if match.groups() else match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                        "level": self._get_structure_level(
                            structure_type, match.group(0)
                        ),
                    }
                )

        # 위치별 정렬
        structure_points.sort(key=lambda x: x["start"])
        return structure_points

    def _get_structure_level(self, structure_type: str, text: str) -> int:
        """구조 요소의 레벨 결정"""
        if structure_type == "title":
            return text.count("#")
        elif structure_type == "section":
            return 2
        elif structure_type == "subsection":
            return 3
        else:
            return 1

    def _split_by_structure(
        self, text: str, structure_points: List[Dict[str, Any]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """구조를 기반으로 텍스트 분할"""
        chunks = []

        for i, point in enumerate(structure_points):
            start = point["start"]
            end = (
                structure_points[i + 1]["start"]
                if i + 1 < len(structure_points)
                else len(text)
            )

            chunk_text = text[start:end].strip()
            section_info = {
                "title": point["text"],
                "level": point["level"],
                "start": start,
                "end": end,
            }

            chunks.append((chunk_text, section_info))

        return chunks

    def _analyze_document(self, text: str) -> Dict[str, Any]:
        """문서 특성 분석"""
        analysis = {
            "has_clear_structure": False,
            "is_conversational": False,
            "is_technical": False,
            "avg_sentence_length": 0,
            "structure_density": 0,
        }

        try:
            # 구조 요소 밀도
            structure_points = self._identify_structure(text)
            analysis["structure_density"] = len(structure_points) / max(
                len(text.split("\n")), 1
            )
            analysis["has_clear_structure"] = analysis["structure_density"] > 0.1

            # 문장 길이 분석
            sentences = self._split_sentences(text)
            if sentences:
                analysis["avg_sentence_length"] = np.mean(
                    [len(s.split()) for s in sentences]
                )

            # 대화형 텍스트 여부
            conversational_indicators = ["질문", "답변", "?", "!", "입니다", "합니다"]
            conversational_count = sum(
                text.count(indicator) for indicator in conversational_indicators
            )
            analysis["is_conversational"] = conversational_count > len(sentences) * 0.3

            # 기술 문서 여부
            technical_indicators = [
                "API",
                "함수",
                "클래스",
                "구현",
                "시스템",
                "알고리즘",
            ]
            technical_count = sum(
                text.count(indicator) for indicator in technical_indicators
            )
            analysis["is_technical"] = technical_count > len(sentences) * 0.1

        except Exception as e:
            logger.warning(f"Document analysis failed: {e}")

        return analysis

    def get_chunking_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """청킹 결과 통계"""
        if not documents:
            return {}

        chunk_sizes = [len(doc.page_content) for doc in documents]
        semantic_scores = [doc.metadata.get("semantic_score", 0) for doc in documents]
        chunk_types = [doc.metadata.get("chunk_type", "unknown") for doc in documents]

        return {
            "total_chunks": len(documents),
            "avg_chunk_size": np.mean(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "avg_semantic_score": np.mean(semantic_scores) if semantic_scores else 0,
            "chunk_types": dict(zip(*np.unique(chunk_types, return_counts=True))),
            "total_text_length": sum(chunk_sizes),
        }
