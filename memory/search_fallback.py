"""
Fallback search strategies for environments without semantic embeddings.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Sequence, TypeVar

from .config import conf

_log = logging.getLogger("walle.search_fallback")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEFAULT_SUFFIX_RULES = ("'s", "ing", "ed", "s")

RowT = TypeVar("RowT")


@dataclass(frozen=True)
class SearchLexicon:
    stopwords: frozenset[str]
    synonyms: dict[str, tuple[str, ...]]


class SearchLexiconRepository:
    """Loads the lexical fallback profile from disk."""

    def __init__(self, path: str):
        self._path = path

    def load(self) -> SearchLexicon:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            _log.warning("Search fallback lexicon not found: %s", self._path)
            return SearchLexicon(stopwords=frozenset(), synonyms={})
        except json.JSONDecodeError as exc:
            _log.warning("Search fallback lexicon is invalid JSON (%s): %s", self._path, exc)
            return SearchLexicon(stopwords=frozenset(), synonyms={})

        stopwords = frozenset(str(word).lower() for word in raw.get("stopwords", []))
        synonyms = {
            str(key).lower(): tuple(str(value).lower() for value in values)
            for key, values in raw.get("synonyms", {}).items()
        }
        return SearchLexicon(stopwords=stopwords, synonyms=synonyms)


@dataclass(frozen=True)
class SearchScore:
    row: RowT
    score: float


class FallbackSearchStrategy(ABC, Generic[RowT]):
    """Strategy interface for lexical fallback search."""

    @abstractmethod
    def score(self, query: str, content: str) -> float:
        raise NotImplementedError

    def rank_rows(
        self,
        query: str,
        rows: Sequence[RowT],
        text_getter: Callable[[RowT], str],
    ) -> list[SearchScore[RowT]]:
        ranked = []
        for row in rows:
            score = self.score(query, text_getter(row))
            if score > 0:
                ranked.append(SearchScore(row=row, score=score))
        return ranked


class TokenExpansionSearchStrategy(FallbackSearchStrategy[RowT]):
    """Token overlap strategy with domain lexicon expansion."""

    def __init__(
        self,
        lexicon: SearchLexicon,
        exact_match_boost: float = 5.0,
        overlap_weight: float = 2.0,
        coverage_weight: float = 1.0,
        suffix_rules: Sequence[str] = _DEFAULT_SUFFIX_RULES,
    ):
        self._lexicon = lexicon
        self._exact_match_boost = exact_match_boost
        self._overlap_weight = overlap_weight
        self._coverage_weight = coverage_weight
        self._suffix_rules = tuple(suffix_rules)

    def score(self, query: str, content: str) -> float:
        normalized_query = self._normalize_text(query)
        normalized_content = self._normalize_text(content)
        if not normalized_query or not normalized_content:
            return 0.0

        query_tokens = self._expand_tokens(self._tokenize(query))
        content_tokens = self._expand_tokens(self._tokenize(content))

        score = 0.0
        if normalized_query in normalized_content:
            score += self._exact_match_boost

        overlap = query_tokens & content_tokens
        if overlap:
            score += len(overlap) * self._overlap_weight
            score += len(overlap) / max(len(query_tokens), 1) * self._coverage_weight

        return score

    def _tokenize(self, text: str) -> list[str]:
        tokens = []
        for raw_token in _TOKEN_RE.findall((text or "").lower()):
            token = self._normalize_token(raw_token)
            if len(token) <= 1 or token in self._lexicon.stopwords:
                continue
            tokens.append(token)
        return tokens

    def _expand_tokens(self, tokens: Iterable[str]) -> set[str]:
        expanded = set(tokens)
        for token in list(expanded):
            expanded.update(self._lexicon.synonyms.get(token, ()))
        return expanded

    def _normalize_text(self, text: str) -> str:
        return " ".join(_TOKEN_RE.findall((text or "").lower()))

    def _normalize_token(self, token: str) -> str:
        normalized = token
        for suffix in self._suffix_rules:
            if suffix == "'s" and normalized.endswith(suffix):
                return normalized[:-2]
            if suffix == "ing" and len(normalized) > 4 and normalized.endswith(suffix):
                return normalized[:-3]
            if suffix == "ed" and len(normalized) > 3 and normalized.endswith(suffix):
                return normalized[:-2]
            if suffix == "s" and len(normalized) > 3 and normalized.endswith(suffix):
                return normalized[:-1]
        return normalized


_default_strategy: TokenExpansionSearchStrategy | None = None


def get_default_fallback_search_strategy() -> TokenExpansionSearchStrategy:
    global _default_strategy
    if _default_strategy is None:
        lexicon = SearchLexiconRepository(conf.SEARCH_FALLBACK_LEXICON_PATH).load()
        _default_strategy = TokenExpansionSearchStrategy(
            lexicon=lexicon,
            exact_match_boost=conf.SEARCH_FALLBACK_EXACT_MATCH_BOOST,
            overlap_weight=conf.SEARCH_FALLBACK_OVERLAP_WEIGHT,
            coverage_weight=conf.SEARCH_FALLBACK_COVERAGE_WEIGHT,
        )
    return _default_strategy
