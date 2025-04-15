# Tokenizer Changelog

[English](./for_tokenizer.md) | [中文](../cn/for_tokenizer.md)

## [v0.3.8] - 2025-04-13
### New Features
- Added support for special characters and emoji tokenization
- Implemented boundary condition test cases

### Bug Fixes
- Fixed tokenization errors with long strings without spaces
- Fixed handling of leading/trailing spaces

### Improvements
- Increased test coverage
- Optimized test environment setup
- Added detailed documentation for test framework architecture
- Standardized performance metrics reporting format

## [v0.3.7] - 2025-04-10
### Test Code Enhancements
- Created comprehensive enhanced test script (test_tokenizer_enhanced.py) with detailed performance benchmarks
- Added boundary condition tests to strengthen error handling and edge case validation
- Implemented performance data collection and visualization features
- Added specialized JSONL processing tests to validate data extraction functionality

## [v0.3.6] - 2025-04-07
### Test Code Optimization
- Refactored tokenizer test scripts for clearer test flow
- Added complete test cases including dictionary creation, training and evaluation
- Optimized test dataset generation and management
- Improved test result presentation and error handling

## [v0.3.5] - 2025-03-25
### Code Optimization
- Merged tokenizer.py and tokenizer_optimized.py to unify tokenizer implementation
- Integrated advantages from both versions to improve code consistency
- Optimized memory usage and performance, reducing resource consumption

### Feature Enhancements
- Improved caching mechanism for better tokenization efficiency
- Enhanced batch processing capability for larger scale text processing
- Optimized error handling and logging for better system stability

## [v0.3.4] - 2025-03-19
### Architecture Optimization
- Enhanced tokenizer module independence, fully taking over tokenization functions
- Optimized tokenizer training process with complete CLI interface
- Improved error handling and progress display

### Feature Enhancement
- Completed tokenizer training, evaluation and saving workflow
- Provided more detailed training status and progress feedback
- Optimized model save path handling logic

## [v0.3.3] - 2025-03-15
### Feature Optimization
- Optimized tokenizer training data preprocessing flow
- Improved tokenizer model save/load mechanism
- Enhanced tokenizer error handling capability

### Bug Fixes
- Fixed memory leaks during training
- Resolved performance bottlenecks with large datasets
- Corrected special character handling in tokenization results

## [v0.3.2] - 2025-03-10
### Feature Enhancement
- Added tokenizer training progress visualization
- Implemented training data validation and cleaning
- Added tokenizer model version management

### Code Optimization
- Refactored core tokenizer algorithm
- Optimized training data loading efficiency
- Improved code documentation and comments

## [v0.3.1] - 2025-03-05
### Feature Improvement
- Added automatic parameter tuning for tokenizer configuration
- Optimized tokenization speed and accuracy
- Added new evaluation metrics for tokenization results

### Bug Fixes
- Fixed race conditions in multi-threaded tokenization
- Resolved rare character tokenization errors
- Corrected model save format issues

## [v0.3.0] - 2025-03-01
### Major Update
- Redesigned tokenizer architecture for better scalability
- Introduced new tokenization algorithms and strategies
- Supported custom dictionaries and rules

### Features
- Implemented deep learning based tokenization model
- Added tokenizer performance test suite
- Supported multiple tokenization mode switching

## [v0.2.1] - 2025-02-20
### Feature Optimization
- Improved tokenization accuracy
- Optimized memory usage efficiency
- Enhanced exception handling mechanism

### Bug Fixes
- Fixed long text tokenization lag
- Resolved tokenization errors in specific scenarios
- Fixed dictionary loading exceptions

## [v0.2.0] - 2025-02-10
### New Features
- Implemented basic tokenization functionality
- Added simple training process
- Supported basic dictionary management

### Basic Architecture
- Built tokenizer core framework
- Implemented basic data processing flow
- Added fundamental test cases

## [v0.1.0] - 2025-02-01
### Initial Version
- Project initialization
- Basic framework setup
- Core interface definition