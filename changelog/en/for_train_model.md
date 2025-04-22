# Training Module (train_model.py) Changelog

[English](./for_train_model.md) | [中文](../cn/for_train_model.md)

---

## [v0.8.6] - 2025-04-10
### Test Code Enhancement
- Created comprehensive test script (test_train_model.py) for model training functionality
- Implemented performance benchmarking for dataset loading with different configurations
- Added utility function testing for memory and time formatting
- Developed visualization testing for training statistics plotting
- Implemented data collection capabilities for performance optimization

## [v0.8.4] - 2025-03-21
### Log System Optimization
- **Log Output Simplification**
  - Removed timestamp and log level from console output to reduce visual noise
  - Preserved complete time and level information in file logs for troubleshooting
  - Refactored log initialization logic to completely resolve log duplication issues
- **Information Grouping and Format Optimization**
  - Adjusted configuration information output format, grouped by functionality
  - Optimized separators and decorative elements while maintaining simplicity
  - Unified log interface, simplified internal implementation
- **Performance Optimization**
  - Reduced log IO operation frequency to improve training performance
  - Optimized logger configuration to avoid resource waste
  - Resolved os module reference conflicts

## [v0.8.3] - 2025-03-20
### User Experience Improvements
- **Progress Bar Optimization**
  - Redesigned progress bar display for more concise and smooth presentation
  - Reduced progress bar refresh rate for better terminal output experience
  - Simplified progress information, keeping only key metrics
  - Unified all progress bar styles for consistency
- **Termination Handling Enhancement**
  - Improved force termination functionality for reliable training stop with Ctrl+C
  - Implemented automatic save mechanism before termination to prevent data loss
  - Added 10-second timeout force quit to avoid hanging situations
  - Optimized user prompts during interruption for better understanding

## [v0.8.2] - 2025-03-18
### Interface and Interaction Optimization
- **Interface Simplification**
  - Redesigned log output format, removed redundant information
  - Simplified progress bar display for clearer training status
  - Removed excessive decorative separators and colored output
  - Unified command-line interface style for consistency with other modules
- **Bug Fixes**
  - Fixed night mode time detection error, resolved datetime module reference issue
  - Optimized logging system to avoid duplicate log output
  - Simplified message output format for better readability
  - Unified message display style

## [v0.8.1] - 2025-03-17
### Emergency Fixes
- **Critical Bug Fixes**
  - Fixed path handling inconsistency by standardizing Path object usage
  - Resolved vocabulary size retrieval error causing model initialization failure
  - Fixed log directory creation logic to ensure correct saving to logs/train_model
  - Resolved delayed termination signal handling
- **Code Quality Improvement**
  - Eliminated duplicate functions and redundant code
  - Standardized exception handling and error log format
  - Improved memory management strategy
  - Optimized checkpoint file naming and management

### Termination Handling Enhancement
- **Responsive Termination**
  - Added batch-level termination response without waiting for epoch completion
  - Implemented layered termination confirmation for data integrity
  - Added user interaction confirmation for emergencies
  - Optimized checkpoint saving logic for quick state preservation

### User Interface Improvements
- **Progress Display Enhancement**
  - Comprehensively beautified training and testing progress bars for better visual feedback
  - Added batch processing time and estimated remaining time display
  - Used colored output to enhance key information visibility
  - Optimized progress bar format for more accurate completion time estimation
- **Time Formatting**
  - Added smart time formatting function with automatic unit adjustment (seconds/minutes/hours)
  - Unified time display format across all locations for consistency
  - Improved time consumption statistics and display method

## [v0.8.0] - 2025-03-16
### Stability and Compatibility Enhancement
- **Data Loading Mechanism Reconstruction**
  - Resolved critical issue of DataLoader worker processes unexpected exit
  - Implemented single-process data loading mode (num_workers=0) for stability
  - Optimized batch processing and memory management for large-scale datasets
  - Added training sample count check and warning mechanism

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