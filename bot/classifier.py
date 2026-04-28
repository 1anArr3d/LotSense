"""
bot/classifier.py — Multilingual intent classifier (serious / lowballer / scammer).

Input:  last ≤3 buyer messages (list[str]), concatenated with </s> separators.
Output: IntentResult(label, confidence, auto_reply)

Gate logic (asymmetric — scam errors cost more than false escalations):
    P(scammer)  >= 0.25  → escalate
    P(serious)  >= 0.85  → auto-reply
    P(lowballer) >= 0.75 → auto-reply
    anything else        → escalate

Model: xlm-roberta-base (English + Spanish, no separate language detection needed).
Falls back to keyword rules until fine-tuned weights exist.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Label = Literal["serious", "lowballer", "scammer"]

_LABELS: list[str] = ["serious", "lowballer", "scammer"]
_MODEL_NAME = "xlm-roberta-base"
_MAX_MESSAGES = 3
_MAX_LENGTH = 128

_SCAM_THRESHOLD = 0.25
_SERIOUS_THRESHOLD = 0.85
_LOWBALL_THRESHOLD = 0.75


@dataclass
class IntentResult:
    label: Label
    confidence: float
    auto_reply: bool


def _gate(probs: dict[str, float]) -> tuple[Label, bool]:
    if probs["scammer"] >= _SCAM_THRESHOLD:
        return "scammer", False
    if probs["serious"] >= _SERIOUS_THRESHOLD:
        return "serious", True
    if probs["lowballer"] >= _LOWBALL_THRESHOLD:
        return "lowballer", True
    return max(probs, key=probs.get), False  # type: ignore[return-value]


# ── Keyword fallback (cold start before any fine-tuned weights exist) ──────

_SCAMMER_RE = re.compile(
    r"\b("
    # payment schemes
    r"zelle|venmo|cashier|western\s*union|wire\s*transfer|"
    r"google\s*pay|apple\s*pay|escrow|money\s*order|"
    # remote buyer tells
    r"ship(ping)?|pick\s*up\s*agent|my\s*(son|daughter|brother|sister|husband|wife)|"
    r"overseas?|out\s*of\s*(state|country|town)|"
    # over-offer / trust-building
    r"full\s*asking|pay\s*more|god\s*bless|blessing|"
    # Spanish equivalents
    r"te\s*env[ií]o\s*(un\s*)?(cheque|pago)|mi\s*(hijo|hija|esposo|esposa)|"
    r"dios\s*te\s*bendiga|mando\s*un\s*cheque|fuera\s*del\s*estado"
    r")\b",
    re.IGNORECASE,
)

_LOWBALLER_RE = re.compile(
    r"\b("
    r"cash\s*today|cash\s*right\s*now|lowest|that'?s?\s*too\s*high|"
    r"best\s*(you\s*can\s*do|price|offer)|final\s*(offer|price)|"
    r"all\s*i\s*(have|got)|won'?t\s*go\s*(higher|more)|take\s*it\s*or\s*leave|"
    r"my\s*budget\s*is|"
    # Spanish equivalents
    r"lo\s*m[áa]s\s*que\s*(puedo|tengo)|es\s*todo\s*lo\s*que\s*tengo|"
    r"[úu]ltimo\s*precio|no\s*puedo\s*m[áa]s"
    r")\b",
    re.IGNORECASE,
)


def _keyword_classify(text: str) -> IntentResult:
    if _SCAMMER_RE.search(text):
        return IntentResult(label="scammer", confidence=0.70, auto_reply=False)
    if _LOWBALLER_RE.search(text):
        return IntentResult(label="lowballer", confidence=0.65, auto_reply=False)
    # Unknown without a model — escalate rather than guess
    return IntentResult(label="serious", confidence=0.50, auto_reply=False)


# ── Classifier ─────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    xlm-roberta-base fine-tuned for 3-class buyer intent classification.
    Bilingual (English / Spanish) — no separate language detection step.
    Uses keyword fallback when weights haven't been trained yet.
    """

    def __init__(self, weights_path: Path | None = None) -> None:
        self._model = None
        self._tokenizer = None

        if weights_path and weights_path.exists():
            self._load(weights_path)

    def _load(self, path: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self._model.eval()
        self._torch = torch

    # ── Inference ──────────────────────────────────────────────────────────

    def classify(self, messages: list[str]) -> IntentResult:
        """
        Classify buyer intent from the last ≤3 messages.
        Older messages provide reclassification context — a lowball opener
        followed by genuine questions should shift the label toward serious.
        """
        window = messages[-_MAX_MESSAGES:]
        text = " </s> ".join(window)

        if self._model is None:
            return _keyword_classify(text)

        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_LENGTH,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs_tensor = torch.softmax(logits, dim=-1)[0]

        probs = {label: float(probs_tensor[i]) for i, label in enumerate(_LABELS)}
        label, auto_reply = _gate(probs)
        return IntentResult(label=label, confidence=probs[label], auto_reply=auto_reply)

    # ── Training ───────────────────────────────────────────────────────────

    def train(
        self,
        examples: list[dict],
        output_path: Path,
        epochs: int = 4,
        batch_size: int = 16,
        lr: float = 2e-5,
    ) -> None:
        """
        Fine-tune on labeled examples and save weights to output_path.
        Each example: {"text": str, "label": "serious"|"lowballer"|"scammer"}
        Minimum ~200 examples per class before results are meaningful.
        """
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        import torch
        from torch.utils.data import Dataset

        label2id = {l: i for i, l in enumerate(_LABELS)}

        class _DS(Dataset):
            def __init__(self, examples: list[dict], tokenizer) -> None:
                self.encodings = tokenizer(
                    [e["text"] for e in examples],
                    truncation=True,
                    padding=True,
                    max_length=_MAX_LENGTH,
                )
                self.labels = [label2id[e["label"]] for e in examples]

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, idx: int) -> dict:
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME,
            num_labels=len(_LABELS),
            id2label={i: l for i, l in enumerate(_LABELS)},
            label2id=label2id,
        )

        args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            save_strategy="no",
            logging_steps=10,
            report_to="none",
        )
        Trainer(model=model, args=args, train_dataset=_DS(examples, tokenizer)).train()
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        self._load(output_path)

    # ── Data helpers ───────────────────────────────────────────────────────

    @staticmethod
    def load_examples(path: Path) -> list[dict]:
        """Load training examples from a JSONL file."""
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
