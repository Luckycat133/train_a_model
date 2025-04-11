# Processor Changelog

[English](./for_processor.md) | [中文](../cn/for_processor.md)

## [v2.3.5] - 2024-04-10
### Test Code Enhancement
- Created comprehensive test script (test_processor.py) for data processing functionality
- Implemented performance benchmarking for batch processing with different configurations
- Added memory pool testing for resource management optimization
- Developed test environment setup and cleanup automation
- Implemented data collection capabilities for performance analysis

## [v2.3.4] - 2024-03-25
### Code Optimization
- Merged tokenizer-related functions to improve code consistency
- Optimized processing flow to reduce redundant operations
- Improved error handling mechanism to enhance system stability

### Feature Enhancement
- Enhanced integration capability with tokenizer module
- Improved data processing efficiency to support larger datasets
- Optimized memory usage to reduce resource consumption

## [v2.3.3] - 2024-03-19
### Architecture Optimization
- Removed tokenizer training related code to achieve separation of responsibilities with tokenizer.py
- Deleted _train_tokenizer method, prompting users to use tokenizer.py directly
- Optimized command-line parameters by removing tokenizer-related options
- Updated documentation and comments to reflect latest architectural changes

### Other Improvements
- Simplified data optimization process description by removing tokenizer migration content
- Optimized code structure to improve module independence