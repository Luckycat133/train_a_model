# Data Module Changelog

[English](./for_data.md) | [中文](../cn/for_data.md)

## [v0.4.2] - 2025-03-13
### Log System Optimization
- Log directory structure reorganization
  - Created dedicated subdirectory: data
  - Reorganized existing log files into corresponding directories
  - Added automatic log directory creation mechanism
- Log management functions
  - Added automatic log archiving and cleanup scripts
  - Implemented module-specific loggers
  - Optimized log file naming rules

## [v0.4.1] - 2025-03-11

## [v0.4.0] - 2025-03-11
### Data Expansion
- Added complete "Complete Song Ci" dataset
  - Contains over 21,000 works from Song Dynasty poets
  - Includes complete metadata of cipai, authors and dynasties
  - All works have punctuation and paragraph optimization
### Data Processing Optimization
- Improved data preprocessing workflow
  - Adapted to new tokenizer's training data format
  - Optimized data cleaning and standardization processes
  - Enhanced data validation and error handling mechanisms
- Data management improvements
  - Optimized data file storage structure
  - Improved data version control
  - Enhanced data integrity checks