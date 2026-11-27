import json

import pytest

from src.dataset import LMDataset
from tokenizer import ClassicalTokenizer


def write_vocab(tmp_path, vocab):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"vocab": vocab}, ensure_ascii=False), encoding="utf-8")
    return path


def test_character_vocab_keeps_known_characters_in_dictionary_phrases(tmp_path):
    tok = ClassicalTokenizer()
    tok.load(str(write_vocab(tmp_path, {"<unk>": 0, "床": 1, "前": 2, "光": 3})))
    assert tok.tokenize("床前光") == ["床前", "光"]
    assert tok.encode("床前光猫") == [1, 2, 3, 0]
    assert tok.decode(tok.encode("床前光")) == "床前光"


def test_exact_vocab_phrase_takes_priority_and_supports_bert_unk(tmp_path):
    tok = ClassicalTokenizer()
    tok.load(str(write_vocab(tmp_path, {"[UNK]": 0, "床前": 1, "床": 2, "前": 3})))
    assert tok.encode("床前猫") == [1, 0]


@pytest.mark.parametrize(
    "vocab", [{"<unk>": 0, "床": True}, {"<unk>": 0, "床": 2}, {"<unk>": 0, "床": 0}]
)
def test_invalid_vocab_is_rejected(tmp_path, vocab):
    with pytest.raises(ValueError, match="vocab"):
        ClassicalTokenizer().load(str(write_vocab(tmp_path, vocab)))


def test_dataset_uses_integer_encode_and_keeps_single_token_records(tmp_path):
    tok = ClassicalTokenizer()
    tok.load(str(write_vocab(tmp_path, {"<unk>": 0, "床": 1, "前": 2, "光": 3})))
    corpus = tmp_path / "sample.jsonl"
    corpus.write_text('{"text":"床前"}\n{"text":"光"}\n', encoding="utf-8")
    dataset = LMDataset(corpus, context_length=2, stride=1, tokenizer=tok)
    assert len(dataset) == 1
    assert dataset.samples[0]["input_ids"].tolist() == [1, 2]
    assert dataset.samples[0]["target_ids"].tolist() == [2, 3]


def test_dataset_rejects_non_integral_ids_from_encoder(tmp_path):
    class InvalidTokenizer:
        def encode(self, text):
            return ["床", "前"]

    corpus = tmp_path / "sample.jsonl"
    corpus.write_text('{"text":"床前"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="integer token IDs"):
        LMDataset(corpus, context_length=2, stride=1, tokenizer=InvalidTokenizer())
