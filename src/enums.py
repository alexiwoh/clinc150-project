"""Project-wide enumerations."""

from __future__ import annotations

from enum import StrEnum


class ModelID(StrEnum):
    """Canonical model identifiers with associated metadata.

    Each member carries a display name and the legacy tuning CSV filename
    from the Steps 4-6 flat layout.  Use ``tuple(ModelID)`` where
    ``CANONICAL_MODEL_IDS`` was previously used.
    """

    def __new__(cls, value: str, display_name: str, legacy_tuning_filename: str) -> ModelID:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._display_name = display_name
        obj._legacy_tuning_filename = legacy_tuning_filename
        return obj

    MLP = ("mlp", "TF-IDF + MLP", "mlp_tuning_results.csv")
    TEXT_CNN = ("text_cnn", "Text CNN", "text_cnn_tuning_results.csv")
    BILSTM = ("bilstm", "BiLSTM", "bilstm_tuning_results.csv")

    @property
    def display_name(self) -> str:
        """Human-readable model name for reports and logs."""
        return self._display_name

    @property
    def legacy_tuning_filename(self) -> str:
        """Tuning CSV filename from the Steps 4-6 flat ``outputs/reports/`` layout."""
        return self._legacy_tuning_filename

    @property
    def input_type(self) -> str:
        """Input representation type used by this model (``tfidf`` or ``token_ids``)."""
        return "tfidf" if self is ModelID.MLP else "token_ids"
