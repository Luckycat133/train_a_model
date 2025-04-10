# Tokenizer Changelog

[English](./for_tokenizer.md) | [中文](../for_tokenizer.md)

## [v0.3.6] - 2024-04-07
### Test Code Optimization
- Refactored tokenizer test scripts for clearer testing process
- Added comprehensive test cases including dictionary creation, training, and evaluation
- Optimized test dataset generation and management
- Improved test result display and error handling

## [v0.3.5] - 2024-03-25
### Code Optimization
- Merged tokenizer.py and tokenizer_optimized.py for unified tokenizer implementation
- Integrated advantages of both versions to improve code consistency
- Optimized memory usage and performance, reduced resource consumption

### Feature Enhancement
- Improved caching mechanism for better tokenization efficiency
- Enhanced batch processing capability for larger text processing
- Optimized error handling and logging for better system stability

## [v0.3.4] - 2024-03-19
### Architecture Optimization
- Enhanced tokenizer module independence, fully managing tokenization-related functions
- Optimized tokenizer training process, providing more complete command-line interface
- Improved error handling and progress display

### Feature Enhancement
- Completed comprehensive workflow for tokenizer training, evaluation, and saving
- Provided more detailed training status and progress feedback
- Optimized model saving path handling logic

## [v0.3.3] - 2024-03-15
### Feature Optimization
- Optimized tokenizer training data preprocessing workflow
- Improved tokenizer model saving and loading mechanism
- Enhanced tokenizer error handling capability

### Bug Fixes
- Fixed memory leak issues during training
- Resolved performance bottlenecks in large-scale dataset training
- Fixed special character handling issues in tokenization results

## [v0.3.2] - 2024-03-10
### Feature Enhancement
- Added tokenizer training progress visualization
- Added training data validation and cleaning functionality
- Implemented tokenizer model version management

### Code Optimization
- Refactored tokenizer core algorithm
- Optimized training data loading efficiency
- Improved code documentation and comments

## [v0.3.1] - 2024-03-05
### Feature Improvements
- Added automatic tuning of tokenizer configuration parameters
- Optimized tokenization speed and accuracy
- Added new tokenization result evaluation metrics

### Bug Fixes
- Fixed race conditions in multi-threaded tokenization
- Resolved rare character tokenization errors
- Fixed model saving format issues

## [v0.3.0] - 2024-03-01
### Major Updates
- Redesigned tokenizer architecture for better extensibility
- Introduced new tokenization algorithms and strategies
- Added support for custom dictionaries and rules

### Features
- Implemented deep learning-based tokenization model
- Added tokenizer performance test suite
- Added support for multiple tokenization modes

## [v0.2.1] - 2024-02-20
### Feature Optimization
- Improved tokenization accuracy
- Optimized memory usage efficiency
- Enhanced exception handling mechanism

### Bug Fixes
- Fixed long text tokenization lag issues
- Resolved tokenization errors in specific scenarios
- Fixed dictionary loading exceptions

## [v0.2.0] - 2024-02-10
### New Features
- Implemented basic tokenization functionality
- Added simple training process
- Added basic dictionary management

### Basic Architecture
- Built tokenizer core framework
- Implemented basic data processing workflow
- Added basic test cases

## [v0.1.0] - 2024-02-01
### Initial Version
- Project initialization
- Basic framework setup
- Core interface definition