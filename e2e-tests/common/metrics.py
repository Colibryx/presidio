"""
Metriche per la valutazione di sistemi di rilevamento e anonimizzazione PII.

Supporta:
- Strict matching: entità deve avere stesso start, end e entity_type
- Partial matching: overlap di caratteri con stesso entity_type
- Precision, Recall, F1, F2 (priorità su recall)
"""

from dataclasses import dataclass, field
from typing import List, Literal

MatchMode = Literal["strict", "partial"]


@dataclass
class PIIEntity:
    """Rappresenta un'entità PII con span e tipo."""

    start: int
    end: int
    entity_type: str
    text: str = ""

    def __hash__(self):
        return hash((self.start, self.end, self.entity_type))

    def overlaps(self, other: "PIIEntity") -> bool:
        """Verifica se c'è overlap tra due span."""
        return not (self.end <= other.start or other.end <= self.start)

    def overlap_ratio(self, other: "PIIEntity") -> float:
        """Rapporto di overlap (intersezione / unione)."""
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        if overlap_start >= overlap_end:
            return 0.0
        intersection = overlap_end - overlap_start
        union = max(self.end, other.end) - min(self.start, other.start)
        return intersection / union if union > 0 else 0.0

    def exact_match(self, other: "PIIEntity") -> bool:
        """Match esatto: stesso start, end e entity_type."""
        return (
            self.start == other.start
            and self.end == other.end
            and self.entity_type == other.entity_type
        )


@dataclass
class EvaluationResult:
    """Risultato delle metriche di valutazione."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    f2: float = 0.0  # F2 dà più peso al recall (importante per PII)
    accuracy: float = 0.0  # (TP + TN) / total, dove TN = non-PII correttamente non rilevato

    def __post_init__(self):
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (
                self.true_positives + self.false_positives
            )
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (
                self.true_positives + self.false_negatives
            )
        if self.precision + self.recall > 0:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
            # F2: β=2, più peso al recall
            self.f2 = (
                5 * self.precision * self.recall / (4 * self.precision + self.recall)
            )


@dataclass
class EntityLevelMetrics:
    """Metriche per singolo tipo di entità."""

    entity_type: str
    result: EvaluationResult
    per_entity_details: dict = field(default_factory=dict)


def _entity_from_dict(d: dict) -> PIIEntity:
    """Converte un dict in PIIEntity."""
    return PIIEntity(
        start=d["start"],
        end=d["end"],
        entity_type=d.get("entity_type", d.get("type", "UNKNOWN")),
        text=d.get("text", ""),
    )


def compute_entity_metrics(
    ground_truth: List[dict],
    predictions: List[dict],
    match_mode: MatchMode = "strict",
) -> tuple[EvaluationResult, List[tuple], List[tuple], List[tuple]]:
    """
    Calcola precision, recall, F1 confrontando ground truth con predictions.

    Args:
        ground_truth: Lista di dict con start, end, entity_type
        predictions: Lista di dict con start, end, entity_type (da analyzer)
        match_mode: "strict" (match esatto) o "partial" (overlap)

    Returns:
        (EvaluationResult, matched_pairs, false_positives, false_negatives)
    """
    gt_entities = [_entity_from_dict(e) for e in ground_truth]
    pred_entities = [_entity_from_dict(e) for e in predictions]

    matched_gt = set()
    matched_pred = set()
    matched_pairs = []

    for pred in pred_entities:
        best_match = None
        best_score = 0.0

        for i, gt in enumerate(gt_entities):
            if i in matched_gt:
                continue
            if pred.entity_type != gt.entity_type:
                continue

            if match_mode == "strict":
                if pred.exact_match(gt):
                    best_match = (i, gt)
                    best_score = 1.0
                    break
            else:  # partial
                if pred.overlaps(gt):
                    score = pred.overlap_ratio(gt)
                    if score > best_score:
                        best_score = score
                        best_match = (i, gt)

        if best_match is not None:
            gt_idx, gt_ent = best_match
            matched_gt.add(gt_idx)
            matched_pred.add(pred_entities.index(pred))
            matched_pairs.append((pred, gt_ent))

    tp = len(matched_pairs)
    fp = len(pred_entities) - tp
    fn = len(gt_entities) - tp

    result = EvaluationResult(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )

    fp_entities = [p for i, p in enumerate(pred_entities) if i not in matched_pred]
    fn_entities = [g for i, g in enumerate(gt_entities) if i not in matched_gt]

    return result, matched_pairs, fp_entities, fn_entities


def compute_anonymization_success(
    original_text: str,
    anonymized_text: str,
    ground_truth: List[dict],
) -> tuple[float, List[dict], List[dict]]:
    """
    Verifica se le entità PII sono state correttamente rimosse dall'output.

    Per ogni entità nel ground truth, controlla se il testo originale
    appare ancora nel testo anonimizzato.

    Returns:
        (recall_anonymization, leaked_entities, correctly_anonymized)
    """
    leaked = []
    correctly_anon = []

    for ent in ground_truth:
        e = _entity_from_dict(ent) if isinstance(ent, dict) else ent
        original_span = original_text[e.start : e.end]
        # Controlla se lo span originale appare nel testo anonimizzato
        if original_span and original_span in anonymized_text:
            leaked.append(
                {
                    "start": e.start,
                    "end": e.end,
                    "entity_type": e.entity_type,
                    "text": original_span,
                }
            )
        else:
            correctly_anon.append(
                {
                    "start": e.start,
                    "end": e.end,
                    "entity_type": e.entity_type,
                    "text": original_span,
                }
            )

    total = len(ground_truth)
    recall = len(correctly_anon) / total if total > 0 else 1.0

    return recall, leaked, correctly_anon


def compute_metrics_by_entity_type(
    ground_truth: List[dict],
    predictions: List[dict],
    match_mode: MatchMode = "strict",
) -> dict[str, EntityLevelMetrics]:
    """Calcola metriche separate per ogni tipo di entità."""
    entity_types = set(
        e.get("entity_type", e.get("type", "UNKNOWN"))
        for e in ground_truth + predictions
    )

    results = {}
    for et in entity_types:
        gt_filtered = [e for e in ground_truth if e.get("entity_type", e.get("type")) == et]
        pred_filtered = [e for e in predictions if e.get("entity_type", e.get("type")) == et]

        result, _, fp_list, fn_list = compute_entity_metrics(
            gt_filtered, pred_filtered, match_mode
        )

        results[et] = EntityLevelMetrics(
            entity_type=et,
            result=result,
            per_entity_details={
                "false_positives": [{"start": e.start, "end": e.end, "text": e.text} for e in fp_list],
                "false_negatives": [{"start": e.start, "end": e.end, "text": e.text} for e in fn_list],
            },
        )

    return results
