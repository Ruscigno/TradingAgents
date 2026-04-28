"""Schema do contrato entre etapas da cascata.

Ver `studies/01-cascata-eliminatoria-de-recomendacoes.md` (seção
"Schema do contrato entre etapas") para o desenho completo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Verdict = Literal["PASS", "NO_PASS"]
StageName = Literal[
    "universe",   # etapa 0: seleção (N/A — não emite verdict por candidata)
    "screener",   # etapa 1: indicadores técnicos determinísticos
    "market",     # etapa 2: Market Analyst
    "analysts",   # etapa 3: Sentiment + News + Fundamentals + composite
    "debate",     # etapa 4: Bull × Bear + judge
    "research",   # etapa 5: Research Manager
    "decision",   # etapa 6: Trader + Risk + Portfolio Manager
]


@dataclass
class Elimination:
    """Registro de descarte de uma candidata em uma etapa."""

    stage: str
    reason: str
    confidence: float


@dataclass
class Candidate:
    """Uma candidata que percorre a cascata.

    Acumula `confidence_history` (uma entrada por etapa que emitiu confidence)
    e `reports` (saídas textuais que alimentam etapas seguintes). Se for
    descartada, `elimination` é preenchido e ela vira `verdict == NO_PASS`.
    """

    ticker: str
    confidence_history: dict[str, float] = field(default_factory=dict)
    reports: dict[str, str] = field(default_factory=dict)
    elimination: Elimination | None = None
    final_score: float | None = None

    @property
    def verdict(self) -> Verdict:
        return "NO_PASS" if self.elimination is not None else "PASS"


@dataclass
class StageOutcome:
    """Resultado de uma etapa: todas as candidatas avaliadas + custo."""

    stage_name: str
    candidates: list[Candidate]
    cost_usd: float = 0.0
    duration_s: float = 0.0

    @property
    def passed(self) -> list[Candidate]:
        return [c for c in self.candidates if c.elimination is None]

    @property
    def eliminated(self) -> list[Candidate]:
        return [c for c in self.candidates if c.elimination is not None]
