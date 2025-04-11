# Training Module (train_model.py) Changelog

## [v0.8.6] - 2024-04-10
### Test Code Enhancement
- Created comprehensive test script (test_train_model.py) for model training functionality
- Implemented performance benchmarking for dataset loading with different configurations
- Added utility function testing for memory and time formatting
- Developed visualization testing for training statistics plotting
- Implemented data collection capabilities for performance optimization

## [v0.8.5] - 2025-03-21
### Global Log Output Optimization
- **Unified System-wide Log Format**
  - Standardized console logs across all modules, removing timestamp and log level prefixes
  - Standardized file log format, preserving complete time, level and module information
  - Optimized log file naming and storage structure
- **Log Processing Mechanism Improvements**
  - Separated console and file log format settings for better user experience
  - Optimized logger configuration method to prevent duplicate log output
  - Enhanced log consistency across different modules

[View in Chinese](../cn/for_train_model.md)