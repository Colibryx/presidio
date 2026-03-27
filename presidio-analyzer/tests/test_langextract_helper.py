"""Tests for llm_utils.langextract_helper module."""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from presidio_analyzer import RecognizerResult
from presidio_analyzer.llm_utils.langextract_helper import (
    extract_lm_config,
    get_supported_entities,
    create_reverse_entity_mapping,
    calculate_extraction_confidence,
    DEFAULT_ALIGNMENT_SCORES,
    convert_langextract_batch_to_presidio_results,
    convert_langextract_batch_to_presidio_results_aligned,
    presidio_langextract_batch_document_id,
)


class TestExtractLmConfig:
    """Tests for extract_lm_config function."""

    def test_when_lm_recognizer_section_exists_then_extracts_all_fields(self):
        """Test extracting all fields from lm_recognizer section."""
        config = {
            "lm_recognizer": {
                "supported_entities": ["PERSON", "EMAIL"],
                "min_score": 0.7,
                "labels_to_ignore": ["system"],
                "enable_generic_consolidation": False
            }
        }

        result = extract_lm_config(config)

        assert result["supported_entities"] == ["PERSON", "EMAIL"]
        assert result["min_score"] == 0.7
        assert result["labels_to_ignore"] == ["system"]
        assert result["enable_generic_consolidation"] is False

    def test_when_lm_recognizer_missing_then_returns_defaults(self):
        """Test that defaults are returned when lm_recognizer section is missing."""
        config = {}

        result = extract_lm_config(config)

        assert result["supported_entities"] is None
        assert result["min_score"] == 0.5
        assert result["labels_to_ignore"] == []
        assert result["enable_generic_consolidation"] is True

    def test_when_partial_config_then_uses_defaults_for_missing_fields(self):
        """Test that defaults are used for missing fields."""
        config = {
            "lm_recognizer": {
                "supported_entities": ["PERSON"]
            }
        }

        result = extract_lm_config(config)

        assert result["supported_entities"] == ["PERSON"]
        assert result["min_score"] == 0.5
        assert result["labels_to_ignore"] == []
        assert result["enable_generic_consolidation"] is True

    def test_when_only_min_score_provided_then_uses_defaults_for_others(self):
        """Test partial config with only min_score."""
        config = {
            "lm_recognizer": {
                "min_score": 0.8
            }
        }

        result = extract_lm_config(config)

        assert result["supported_entities"] is None
        assert result["min_score"] == 0.8
        assert result["labels_to_ignore"] == []
        assert result["enable_generic_consolidation"] is True


class TestGetSupportedEntities:
    """Tests for get_supported_entities function."""

    def test_when_lm_config_has_entities_then_returns_from_lm_config(self):
        """Test that entities from lm_config are prioritized."""
        lm_config = {"supported_entities": ["PERSON", "EMAIL"]}
        langextract_config = {"supported_entities": ["PHONE"]}

        result = get_supported_entities(lm_config, langextract_config)

        assert result == ["PERSON", "EMAIL"]

    def test_when_lm_config_missing_entities_then_returns_from_langextract_config(self):
        """Test fallback to langextract_config when lm_config has no entities."""
        lm_config = {"supported_entities": None}
        langextract_config = {"supported_entities": ["PHONE", "ADDRESS"]}

        result = get_supported_entities(lm_config, langextract_config)

        assert result == ["PHONE", "ADDRESS"]

    def test_when_both_missing_entities_then_returns_none(self):
        """Test that None is returned when both configs lack entities."""
        lm_config = {}
        langextract_config = {}

        result = get_supported_entities(lm_config, langextract_config)

        assert result is None

    def test_when_lm_config_has_empty_list_then_falls_back_to_langextract(self):
        """Test that empty list in lm_config causes fallback."""
        lm_config = {"supported_entities": []}
        langextract_config = {"supported_entities": ["PERSON"]}

        result = get_supported_entities(lm_config, langextract_config)

        # Empty list is falsy, so should fallback
        assert result == ["PERSON"]


class TestCreateReverseEntityMapping:
    """Tests for create_reverse_entity_mapping function."""

    def test_when_mapping_provided_then_creates_reverse(self):
        """Test creating reverse mapping from entity mappings."""
        entity_mappings = {
            "person": "PERSON",
            "email": "EMAIL_ADDRESS",
            "phone": "PHONE_NUMBER"
        }

        result = create_reverse_entity_mapping(entity_mappings)

        assert result["PERSON"] == "person"
        assert result["EMAIL_ADDRESS"] == "email"
        assert result["PHONE_NUMBER"] == "phone"

    def test_when_empty_mapping_then_returns_empty_dict(self):
        """Test that empty mapping returns empty dict."""
        result = create_reverse_entity_mapping({})

        assert result == {}

    def test_when_duplicate_values_then_last_wins(self):
        """Test behavior with duplicate values in original mapping."""
        entity_mappings = {
            "person1": "PERSON",
            "person2": "PERSON"  # Duplicate value
        }

        result = create_reverse_entity_mapping(entity_mappings)

        # Last key-value pair should win
        assert result["PERSON"] == "person2"


class TestCalculateExtractionConfidence:
    """Tests for calculate_extraction_confidence function."""

    def test_when_extraction_has_match_exact_then_returns_095(self):
        """Test confidence for MATCH_EXACT alignment."""
        extraction = Mock()
        extraction.alignment_status = "MATCH_EXACT"

        result = calculate_extraction_confidence(extraction)

        assert result == 0.95

    def test_when_extraction_has_match_fuzzy_then_returns_080(self):
        """Test confidence for MATCH_FUZZY alignment."""
        extraction = Mock()
        extraction.alignment_status = "MATCH_FUZZY"

        result = calculate_extraction_confidence(extraction)

        assert result == 0.80

    def test_when_extraction_has_match_lesser_then_returns_070(self):
        """Test confidence for MATCH_LESSER alignment."""
        extraction = Mock()
        extraction.alignment_status = "MATCH_LESSER"

        result = calculate_extraction_confidence(extraction)

        assert result == 0.70

    def test_when_extraction_has_not_aligned_then_returns_060(self):
        """Test confidence for NOT_ALIGNED alignment."""
        extraction = Mock()
        extraction.alignment_status = "NOT_ALIGNED"

        result = calculate_extraction_confidence(extraction)

        assert result == 0.60

    def test_when_extraction_missing_alignment_status_then_returns_default(self):
        """Test that default score is returned when alignment_status is missing."""
        extraction = Mock(spec=[])  # Mock without alignment_status attribute

        result = calculate_extraction_confidence(extraction)

        assert result == 0.85

    def test_when_alignment_status_is_none_then_returns_default(self):
        """Test that default score is returned when alignment_status is None."""
        extraction = Mock()
        extraction.alignment_status = None

        result = calculate_extraction_confidence(extraction)

        assert result == 0.85

    def test_when_alignment_status_is_empty_string_then_returns_default(self):
        """Test that default score is returned for empty alignment_status."""
        extraction = Mock()
        extraction.alignment_status = ""

        result = calculate_extraction_confidence(extraction)

        assert result == 0.85

    def test_when_unknown_alignment_status_then_returns_default(self):
        """Test that default score is returned for unknown alignment status."""
        extraction = Mock()
        extraction.alignment_status = "UNKNOWN_STATUS"

        result = calculate_extraction_confidence(extraction)

        assert result == 0.85

    def test_when_alignment_status_lowercase_then_matches_correctly(self):
        """Test that alignment status matching is case-insensitive."""
        extraction = Mock()
        extraction.alignment_status = "match_exact"  # lowercase

        result = calculate_extraction_confidence(extraction)

        # Should still match because we convert to uppercase
        assert result == 0.95

    def test_when_custom_alignment_scores_provided_then_uses_custom(self):
        """Test using custom alignment scores."""
        extraction = Mock()
        extraction.alignment_status = "MATCH_EXACT"
        
        custom_scores = {
            "MATCH_EXACT": 0.99,
            "MATCH_FUZZY": 0.75
        }

        result = calculate_extraction_confidence(extraction, alignment_scores=custom_scores)

        assert result == 0.99

    def test_when_custom_scores_missing_status_then_returns_default(self):
        """Test that default is returned when custom scores don't have the status."""
        extraction = Mock()
        extraction.alignment_status = "MATCH_FUZZY"
        
        custom_scores = {
            "MATCH_EXACT": 0.99
            # MATCH_FUZZY missing
        }

        result = calculate_extraction_confidence(extraction, alignment_scores=custom_scores)

        assert result == 0.85  # Default score


class TestConvertLangextractBatchToPresidioResults:
    """Tests for convert_langextract_batch_to_presidio_results."""

    def test_when_batch_is_list_of_documents_then_returns_per_document_lists(self):
        doc1_extraction = Mock()
        doc1_extraction.extraction_class = "person"
        doc1_extraction.char_interval = Mock(start_pos=0, end_pos=4)
        doc1_extraction.alignment_status = "MATCH_EXACT"
        doc1_extraction.attributes = {}

        doc2_extraction = Mock()
        doc2_extraction.extraction_class = "email"
        doc2_extraction.char_interval = Mock(start_pos=1, end_pos=5)
        doc2_extraction.alignment_status = "MATCH_EXACT"
        doc2_extraction.attributes = {}

        r1 = Mock()
        r1.extractions = [doc1_extraction]
        r2 = Mock()
        r2.extractions = [doc2_extraction]

        entity_mappings = {"person": "PERSON", "email": "EMAIL_ADDRESS"}
        out = convert_langextract_batch_to_presidio_results(
            langextract_batch_result=[r1, r2],
            entity_mappings=entity_mappings,
            supported_entities=["PERSON", "EMAIL_ADDRESS", "GENERIC_PII_ENTITY"],
            enable_generic_consolidation=True,
            recognizer_name="Test",
        )

        assert len(out) == 2
        assert len(out[0]) == 1
        assert isinstance(out[0][0], RecognizerResult)
        assert out[0][0].entity_type == "PERSON"
        assert len(out[1]) == 1
        assert out[1][0].entity_type == "EMAIL_ADDRESS"

    def test_when_batch_result_has_documents_attr_then_uses_it(self):
        inner1 = Mock()
        inner1.extractions = []
        inner2 = Mock()
        inner2.extractions = []
        wrapper = Mock()
        wrapper.documents = [inner1, inner2]

        out = convert_langextract_batch_to_presidio_results(
            langextract_batch_result=wrapper,
            entity_mappings={"person": "PERSON"},
            supported_entities=["PERSON", "GENERIC_PII_ENTITY"],
            enable_generic_consolidation=True,
            recognizer_name="Test",
        )
        assert out == [[], []]


class TestConvertLangextractBatchAligned:
    """Tests for convert_langextract_batch_to_presidio_results_aligned."""

    @staticmethod
    def _mock_doc(document_id, start_pos):
        ext = Mock()
        ext.extraction_class = "person"
        ext.char_interval = Mock(start_pos=start_pos, end_pos=start_pos + 1)
        ext.alignment_status = "MATCH_EXACT"
        ext.attributes = {}
        d = Mock()
        d.document_id = document_id
        d.extractions = [ext]
        return d

    def test_reorders_by_document_id_when_langextract_order_differs(self):
        docs = [
            self._mock_doc(presidio_langextract_batch_document_id(1), 99),
            self._mock_doc(presidio_langextract_batch_document_id(0), 5),
        ]
        out = convert_langextract_batch_to_presidio_results_aligned(
            docs,
            num_input_documents=2,
            entity_mappings={"person": "PERSON"},
            supported_entities=["PERSON", "GENERIC_PII_ENTITY"],
            enable_generic_consolidation=True,
            recognizer_name="Test",
        )
        assert len(out) == 2
        assert out[0][0].start == 5
        assert out[1][0].start == 99

    def test_pads_missing_documents_with_empty_lists(self):
        d0 = self._mock_doc(presidio_langextract_batch_document_id(0), 1)
        out = convert_langextract_batch_to_presidio_results_aligned(
            [d0],
            num_input_documents=2,
            entity_mappings={"person": "PERSON"},
            supported_entities=["PERSON", "GENERIC_PII_ENTITY"],
            enable_generic_consolidation=True,
            recognizer_name="Test",
        )
        assert len(out) == 2
        assert len(out[0]) == 1
        assert out[1] == []


class TestDefaultAlignmentScores:
    """Tests for DEFAULT_ALIGNMENT_SCORES constant."""

    def test_default_alignment_scores_has_all_statuses(self):
        """Test that DEFAULT_ALIGNMENT_SCORES contains expected statuses."""
        assert "MATCH_EXACT" in DEFAULT_ALIGNMENT_SCORES
        assert "MATCH_FUZZY" in DEFAULT_ALIGNMENT_SCORES
        assert "MATCH_LESSER" in DEFAULT_ALIGNMENT_SCORES
        assert "NOT_ALIGNED" in DEFAULT_ALIGNMENT_SCORES

    def test_default_alignment_scores_values_are_valid(self):
        """Test that all scores are between 0 and 1."""
        for status, score in DEFAULT_ALIGNMENT_SCORES.items():
            assert 0.0 <= score <= 1.0, f"{status} score {score} is not between 0 and 1"

    def test_default_alignment_scores_are_ordered_correctly(self):
        """Test that scores are in descending order of confidence."""
        assert DEFAULT_ALIGNMENT_SCORES["MATCH_EXACT"] > DEFAULT_ALIGNMENT_SCORES["MATCH_FUZZY"]
        assert DEFAULT_ALIGNMENT_SCORES["MATCH_FUZZY"] > DEFAULT_ALIGNMENT_SCORES["MATCH_LESSER"]
        assert DEFAULT_ALIGNMENT_SCORES["MATCH_LESSER"] > DEFAULT_ALIGNMENT_SCORES["NOT_ALIGNED"]


class TestPatchOpenaiWithLangfuse:
    """Tests for _patch_openai_with_langfuse function."""

    @patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"})
    @patch("presidio_analyzer.llm_utils.langextract_helper.LANGFUSE_AVAILABLE", True)
    def test_when_langfuse_available_and_configured_then_patches_provider(self):
        """Test that openai module in langextract provider is replaced with langfuse wrapper."""
        mock_langfuse_openai = MagicMock()
        mock_provider_module = MagicMock()

        with patch(
            "presidio_analyzer.llm_utils.langextract_helper._langfuse_openai",
            mock_langfuse_openai,
        ):
            with patch.dict(
                "sys.modules",
                {"langextract.providers.openai": mock_provider_module},
            ):
                from presidio_analyzer.llm_utils.langextract_helper import (
                    _patch_openai_with_langfuse,
                )
                _patch_openai_with_langfuse()

                assert mock_provider_module.openai is mock_langfuse_openai

    @patch.dict(os.environ, {}, clear=True)
    @patch("presidio_analyzer.llm_utils.langextract_helper.LANGFUSE_AVAILABLE", False)
    def test_when_langfuse_not_available_then_does_not_patch(self):
        """Test that no patching occurs when langfuse is not installed."""
        mock_provider_module = MagicMock()
        original_openai = mock_provider_module.openai

        with patch.dict(
            "sys.modules",
            {"langextract.providers.openai": mock_provider_module},
        ):
            from presidio_analyzer.llm_utils.langextract_helper import (
                _patch_openai_with_langfuse,
            )
            _patch_openai_with_langfuse()

            assert mock_provider_module.openai is original_openai

    @patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk-test"}, clear=True)
    @patch("presidio_analyzer.llm_utils.langextract_helper.LANGFUSE_AVAILABLE", True)
    def test_when_langfuse_secret_key_missing_then_does_not_patch(self):
        """Test that no patching occurs when LANGFUSE_SECRET_KEY is missing."""
        mock_provider_module = MagicMock()
        original_openai = mock_provider_module.openai

        with patch.dict(
            "sys.modules",
            {"langextract.providers.openai": mock_provider_module},
        ):
            from presidio_analyzer.llm_utils.langextract_helper import (
                _patch_openai_with_langfuse,
            )
            _patch_openai_with_langfuse()

            assert mock_provider_module.openai is original_openai

    @patch.dict(os.environ, {"LANGFUSE_SECRET_KEY": "sk-test"}, clear=True)
    @patch("presidio_analyzer.llm_utils.langextract_helper.LANGFUSE_AVAILABLE", True)
    def test_when_langfuse_public_key_missing_then_does_not_patch(self):
        """Test that no patching occurs when LANGFUSE_PUBLIC_KEY is missing."""
        mock_provider_module = MagicMock()
        original_openai = mock_provider_module.openai

        with patch.dict(
            "sys.modules",
            {"langextract.providers.openai": mock_provider_module},
        ):
            from presidio_analyzer.llm_utils.langextract_helper import (
                _patch_openai_with_langfuse,
            )
            _patch_openai_with_langfuse()

            assert mock_provider_module.openai is original_openai
