# Processor Changelog

## [v2.4.0] - 2024-04-18
### ✨ Feature Enhancement (Pre-training Data Quality Improvement)

- **Logging System Optimization**: Refactored logging system with color differentiation and more detailed timestamp format.
- **Error Handling Improvement**: Enhanced error handling mechanism with more context information for debugging.
- **Performance Optimization**: Optimized batch processing flow to reduce memory usage.

## [v2.4.0] - 2024-04-17
### ✨ Feature Enhancement (Pre-training Data Quality Improvement)

- **Implemented PII Redaction**:
    - Integrated `presidio` library (`presidio-analyzer`, `presidio-anonymizer`) for detecting and redacting Personal Identifiable Information.
    - Replaced the placeholder `pii_redactor` function in `processor.py` with the actual implementation using Presidio.
    - Utilized `spacy` models (e.g., `en_core_web_lg`) to support PII detection.
    - Added PII redaction configurations (enable toggle, entity list, redaction method, replacement tag, Spacy model) to `config/config.yaml` and command-line arguments.
    - Updated `requirements.txt` to include `presidio-analyzer`, `presidio-anonymizer`, and `spacy`.
- **Implemented Harmful Content Filtering (Infrastructure)**:
    - Added `check_harmful_content` function in `processor.py` to determine if text is harmful (currently includes placeholder logic, pending model/API integration).
    - Incorporated this filtering step into the `clean_text` workflow.
    - Added harmful content filtering configurations (enable toggle, model path, confidence threshold) to `config/config.yaml` and command-line arguments.
    - Updated `requirements.txt` to include `transformers` (if not already present or version needs update).
- **Implemented Enhanced Quality Filtering**:
    - Added more sophisticated quality filtering rules within the `clean_text` function in `processor.py`.
    - **Length Filtering**: Added a maximum length limit (`max_length`).
    - **Language Detection**: Use `langdetect` or `fasttext` (optional) to filter out text not in specified languages.
    - **Repetition Detection**: Calculate n-gram repetition rates to filter out overly repetitive text.
    - Added relevant configurations (enable toggle, min/max length, symbol ratio, language filter toggle, allowed languages, detection method, fasttext model path, repetition filter toggle, n-gram size, repetition threshold) to `config/config.yaml` and command-line arguments.
    - Updated `requirements.txt` to include `langdetect` and `fasttext-wheel`.

### 🗑️ Removed Features
    - Removed harmful content filtering functionality and related configurations.
- **Dependencies & Configuration**:
    - Updated `requirements.txt` to include all new dependencies (`chardet`, `beautifulsoup4`, `colorlog`, `inquirer` were also added for other functionalities).
    - Updated `config/config.yaml` by adding a `cleaning_rules` section to centralize all cleaning and filtering options.
    - Refined the logic in `processor.py` for merging command-line arguments and configuration file settings.

## [v2.3.7] - Planned
### Feature Enhancement (Pre-training Data Quality Improvement)

- **Personal Identifiable Information (PII) Redaction**: Plan to add mechanisms to detect and remove/replace sensitive personal information (e.g., names, addresses, phone numbers).
- **Harmful Content Filtering**: Plan to integrate functionality to identify and filter out harmful, inappropriate, or offensive content.
- **Enhanced Quality Filtering**: Plan to introduce more refined quality filtering rules (e.g., based on sentence length, symbol ratio, repetition ratio) or models to filter low-quality text.
- **Punctuation Handling Adjustment**: Plan to review and potentially adjust the punctuation handling logic, leaning towards preserving punctuation crucial for semantic understanding.

## [v2.3.6] - 2025-04-11
### Feature Optimization & User Experience Enhancement

- **Command-Line Arguments**: Improved `argparse` help messages using `ArgumentDefaultsHelpFormatter` and argument groups for better clarity.
- **Progress Bar**: Enhanced `tqdm` progress bar format (`bar_format`) to provide more detailed feedback (e.g., remaining time, processing rate) during file reading and batch processing.
- **Error Handling**: Improved error logging during file reading and preview, now including specific exception types for easier debugging.
- **Logging Setup**: Made the `setup_logging` function more robust by adding exception handling and logging configuration details.

[English](./for_processor.md) | [中文](../cn/for_processor.md)

## [v2.3.5] - 2025-04-10
### Test Code Enhancement
- Created comprehensive test script (test_processor.py) for data processing functionality
- Implemented performance benchmarking for batch processing with different configurations
- Added memory pool testing for resource management optimization
- Developed test environment setup and cleanup automation
- Implemented data collection capabilities for performance analysis

## [v2.3.4] - 2025-03-25
### Code Optimization
- Merged tokenizer-related functions to improve code consistency
- Optimized processing flow to reduce redundant operations
- Improved error handling mechanism to enhance system stability

### Feature Enhancement
- Enhanced integration capability with tokenizer module
- Improved data processing efficiency to support larger datasets
- Optimized memory usage to reduce resource consumption

## [v2.3.3] - 2025-03-19
### Architecture Optimization
- Removed tokenizer training related code to achieve separation of responsibilities with tokenizer.py
- Deleted _train_tokenizer method, prompting users to use tokenizer.py directly
- Optimized command-line parameters by removing tokenizer-related options
- Updated documentation and comments to reflect latest architectural changes

### Other Improvements
- Simplified data optimization process description by removing tokenizer migration content
- Optimized code structure to improve module independence

## [v2.3.2] - 2025-03-15
### Performance Optimization
- Optimized data processing flow to improve speed
- Improved memory management mechanism to reduce resource usage
- Implemented batch processing for large-scale data handling

### Feature Enhancement
- Added data validation and cleaning functions
- Implemented processing progress visualization
- Supported custom processing rules

## [v2.3.1] - 2025-03-10
### Stability Improvement
- Fixed race conditions in concurrent processing
- Resolved memory overflow in large file processing
- Enhanced error handling mechanism

### Functional Optimization
- Optimized data preprocessing pipeline
- Improved text cleaning algorithms
- Strengthened anomaly detection capabilities

## [v2.3.0] - 2025-03-05
### Major Update
- Refactored core data processing architecture
- Implemented modular processing workflow
- Added plugin system support

### New Features
- Created data processing pipelines
- Added task scheduling system
- Supported custom processor extensions

## [v2.2.0] - 2025-02-20
### Feature Enhancement
- Added data statistical analysis capabilities
- Implemented processing result visualization
- Supported additional data export formats

### System Optimization
- Improved configuration management system
- Optimized logging mechanism
- Enhanced processor performance

## [v2.1.0] - 2025-02-10
### New Features
- Implemented automated processing workflow
- Added batch processing support
- Created processing template system

### Improvements
- Optimized processing algorithms
- Improved memory utilization efficiency
- Enhanced error recovery capabilities

## [v2.0.0] - 2025-02-01
### Architecture Refactoring
- New processor architecture design
- Improved data processing workflow
- Optimized core algorithm implementation

### Key Features
- Supported multiple data sources
- Implemented data transformation functions
- Added data validation system

## [v1.0.0] - 2025-01-20
### Initial Release
- Basic data processing functionality
- Simple text processing support
- Fundamental error handling mechanisms