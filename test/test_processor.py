import pytest
from pathlib import Path
from processors.processor import load_config, detect_encoding, clean_text, read_data

@pytest.fixture
def sample_config(tmp_path):
    config_content = '''
paths:
  input_dir: test_input
  output_dir: test_output
processor:
  batch_size: 500
cleaning_rules:
  remove_html: true
  normalize_whitespace: false
    '''
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file

@pytest.fixture
def sample_files(tmp_path):
    test_dir = tmp_path / "test_input"
    test_dir.mkdir()
    
    # 创建不同编码的测试文件
    files = {
        "utf8.txt": "你好，世界！".encode('utf-8'),
        "gbk.txt": "你好，世界！".encode('gbk'),
        "empty.txt": b""
    }
    
    for name, content in files.items():
        (test_dir / name).write_bytes(content)
    
    return test_dir

def test_load_config(sample_config):
    config = load_config(sample_config)
    assert config['paths']['input_dir'] == "test_input"
    assert config['processor']['batch_size'] == 500
    assert config['cleaning_rules']['remove_html'] is True

def test_detect_encoding(sample_files):
    utf8_file = sample_files / "utf8.txt"
    assert detect_encoding(str(utf8_file)) == 'utf-8'
    
    gbk_file = sample_files / "gbk.txt"
    assert detect_encoding(str(gbk_file)) in ['GB2312', 'GBK']
    
    empty_file = sample_files / "empty.txt"
    assert detect_encoding(str(empty_file)) == 'utf-8'

def test_clean_text():
    test_cases = [
        ("<p>Hello</p>", "Hello"),
        ("Hello\n\tWorld", "Hello World"),
        ("测试\x08控制字符", "测试控制字符"),
        ("https://example.com", "")
    ]
    
    cleaning_rules = {
        "remove_html": True,
        "normalize_whitespace": True,
        "remove_control_chars": True,
        "remove_urls": True
    }
    
    for input_text, expected in test_cases:
        assert clean_text(input_text, cleaning_rules) == expected

def test_read_data(sample_files):
    file_list = [str(sample_files / "utf8.txt"), str(sample_files / "gbk.txt")]
    data_gen = read_data(file_list, [".txt"], batch_size=2, preview=False)
    
    batches = list(data_gen)
    assert len(batches) == 1
    assert len(batches[0]) == 2
    assert "你好，世界！" in batches[0]

@pytest.mark.parametrize("text,expected", [
    ("A" * 5, ""),    # 太短
    ("A" * 15 + "!@#$" * 5, ""),  # 符号比例过高
    ("Valid text with normal length", "Valid text with normal length")
])
def test_quality_filtering(text, expected):
    rules = {
        "filter_quality": True,
        "min_length": 10,
        "max_symbol_ratio": 0.2
    }
    assert clean_text(text, rules) == expected