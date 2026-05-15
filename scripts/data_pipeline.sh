#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/dataset"
LOG_DIR="$PROJECT_ROOT/logs"

RAW_DIR="$DATA_DIR/raw"
PROCESSED_DIR="$DATA_DIR/processed"
TOKENIZED_DIR="$DATA_DIR/tokenized"

DOWNLOAD_SOURCES=()
WORKERS=1
VERBOSE=false
DRY_RUN=false
FORCE=false

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] [$level] $message"
}

info() {
    log "INFO" "$@"
}

warn() {
    log "WARN" "$@" >&2
}

error() {
    log "ERROR" "$@" >&2
    exit 1
}

create_directories() {
    info "Creating directory structure..."
    mkdir -p "$RAW_DIR"
    mkdir -p "$PROCESSED_DIR"
    mkdir -p "$TOKENIZED_DIR"
    mkdir -p "$LOG_DIR"
    info "Directory structure created successfully"
}

check_dependencies() {
    info "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    if ! python3 -c "import json" 2>/dev/null; then
        missing_deps+=("json (python)")
    fi
    
    if ! python3 -c "import chardet" 2>/dev/null; then
        missing_deps+=("chardet (python)")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}. Please install them first."
    fi
    
    info "All dependencies are available"
}

download_data() {
    local source="$1"
    info "Downloading data from source: $source"
    
    case "$source" in
        poetry)
            download_poetry
            ;;
        classical)
            download_classical
            ;;
        terms)
            download_terms
            ;;
        all)
            download_poetry
            download_classical
            download_terms
            ;;
        *)
            warn "Unknown source: $source, skipping..."
            ;;
    esac
    
    info "Download from $source completed"
}

download_poetry() {
    info "Downloading poetry data..."
    
    if [ -f "$PROJECT_ROOT/processors/download_poetry.py" ]; then
        python3 "$PROJECT_ROOT/processors/download_poetry.py" || {
            warn "Built-in download script failed, creating sample data..."
            create_sample_poetry_data
        }
    else
        warn "Download script not found, creating sample data..."
        create_sample_poetry_data
    fi
}

download_classical() {
    info "Downloading classical texts..."
    
    if [ -f "$DATA_DIR/classical.jsonl" ]; then
        info "Classical data already exists at $DATA_DIR/classical.jsonl"
        if [ "$FORCE" = false ]; then
            info "Use --force to overwrite existing data"
            return 0
        fi
    fi
    
    create_sample_classical_data
}

download_terms() {
    info "Downloading classical terms..."
    
    if [ -f "$PROCESSED_DIR/terms.jsonl" ]; then
        info "Terms data already exists"
        if [ "$FORCE" = false ]; then
            return 0
        fi
    fi
    
    create_sample_terms_data
}

create_sample_poetry_data() {
    info "Creating sample poetry data..."
    
    cat > "$PROCESSED_DIR/poetry.jsonl" << 'EOF'
{"book": "唐诗三百首", "chapter": "五言绝句", "paragraph_index": 1, "title": "静夜思", "author": "李白", "dynasty": "唐", "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。", "difficulty": 0.2, "tags": ["五言绝句", "思乡", "月夜", "李白"]}
{"book": "唐诗三百首", "chapter": "五言绝句", "paragraph_index": 2, "title": "登鹳雀楼", "author": "王之涣", "dynasty": "唐", "content": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。", "difficulty": 0.2, "tags": ["五言绝句", "登楼", "黄河", "王之涣"]}
{"book": "唐诗三百首", "chapter": "七言绝句", "paragraph_index": 1, "title": "望庐山瀑布", "author": "李白", "dynasty": "唐", "content": "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。", "difficulty": 0.25, "tags": ["七言绝句", "山水", "瀑布", "李白"]}
{"book": "宋词精选", "chapter": "豪放词", "paragraph_index": 1, "title": "念奴娇·赤壁怀古", "author": "苏轼", "dynasty": "宋", "content": "大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。", "difficulty": 0.55, "tags": ["豪放词", "怀古", "赤壁", "苏轼"]}
EOF
    
    info "Sample poetry data created at $PROCESSED_DIR/poetry.jsonl"
}

create_sample_classical_data() {
    info "Creating sample classical texts data..."
    
    cat > "$DATA_DIR/classical.jsonl" << 'EOF'
{"book": "论语", "chapter": "学而", "paragraph_index": 1, "content": "子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？", "difficulty": 0.4, "tags": ["论语", "儒家", "修身"]}
{"book": "论语", "chapter": "为政", "paragraph_index": 1, "content": "子曰：为政以德，譬如北辰，居其所而众星共之。", "difficulty": 0.45, "tags": ["论语", "儒家", "政治"]}
{"book": "道德经", "chapter": "第一章", "paragraph_index": 1, "content": "道可道，非常道。名可名，非常名。无名天地之始；有名万物之母。", "difficulty": 0.7, "tags": ["道德经", "道家", "哲学", "玄学"]}
{"book": "庄子", "chapter": "逍遥游", "paragraph_index": 1, "content": "北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏。", "difficulty": 0.65, "tags": ["庄子", "道家", "寓言"]}
EOF
    
    info "Sample classical data created at $DATA_DIR/classical.jsonl"
}

create_sample_terms_data() {
    info "Creating sample terms data..."
    
    cat > "$PROCESSED_DIR/terms.jsonl" << 'EOF'
{"book": "中医术语", "chapter": "基础理论", "paragraph_index": 1, "content": "阴阳", "difficulty": 0.3, "tags": ["中医", "基础概念"]}
{"book": "中医术语", "chapter": "基础理论", "paragraph_index": 2, "content": "五行", "difficulty": 0.35, "tags": ["中医", "基础概念"]}
{"book": "天文术语", "chapter": "星象", "paragraph_index": 1, "content": "二十八宿", "difficulty": 0.5, "tags": ["天文", "星象"]}
EOF
    
    info "Sample terms data created at $PROCESSED_DIR/terms.jsonl"
}

clean_data() {
    local input_file="$1"
    local output_file="$2"
    
    info "Cleaning data: $input_file -> $output_file"
    
    if [ ! -f "$input_file" ]; then
        error "Input file not found: $input_file"
    fi
    
    python3 << PYTHON_SCRIPT
import json
import re
import sys

input_file = "$input_file"
output_file = "$output_file"

html_pattern = re.compile(r'<[^>]+>')
url_pattern = re.compile(r'https?://\S+')
control_pattern = re.compile(r'[\x00-\x1F\x7F]')
whitespace_pattern = re.compile(r'\s+')

cleaned_count = 0
total_count = 0

with open(input_file, 'r', encoding='utf-8') as fin:
    with open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_count += 1
            try:
                record = json.loads(line.strip())
                
                if 'content' not in record:
                    continue
                
                content = record['content']
                
                content = html_pattern.sub('', content)
                content = url_pattern.sub('', content)
                content = control_pattern.sub('', content)
                content = whitespace_pattern.sub(' ', content).strip()
                
                if len(content) < 5:
                    continue
                
                record['content'] = content
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                cleaned_count += 1
                
            except json.JSONDecodeError:
                continue

print(f"Cleaned: {cleaned_count}/{total_count} records")
PYTHON_SCRIPT
    
    info "Data cleaning completed: $cleaned_count records"
}

convert_format() {
    local input_file="$1"
    local output_file="$2"
    local target_format="${3:-jsonl}"
    
    info "Converting format: $input_file -> $output_file (format: $target_format)"
    
    if [ ! -f "$input_file" ]; then
        error "Input file not found: $input_file"
    fi
    
    case "$target_format" in
        jsonl)
            if [ "$input_file" != "$output_file" ]; then
                cp "$input_file" "$output_file"
            fi
            ;;
        json)
            python3 << PYTHON_SCRIPT
import json

input_file = "$input_file"
output_file = "$output_file"

records = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Converted {len(records)} records to JSON array format")
PYTHON_SCRIPT
            ;;
        *)
            error "Unsupported target format: $target_format"
            ;;
    esac
    
    info "Format conversion completed"
}

merge_datasets() {
    local output_file="${1:-$PROCESSED_DIR/merged.jsonl}"
    shift
    local input_files=("$@")
    
    info "Merging datasets into: $output_file"
    
    if [ ${#input_files[@]} -eq 0 ]; then
        input_files=("$PROCESSED_DIR/poetry.jsonl" "$DATA_DIR/classical.jsonl" "$PROCESSED_DIR/terms.jsonl")
    fi
    
    local total_records=0
    
    > "$output_file"
    
    for input_file in "${input_files[@]}"; do
        if [ -f "$input_file" ]; then
            local file_records=$(wc -l < "$input_file")
            info "Adding $input_file ($file_records records)"
            cat "$input_file" >> "$output_file"
            total_records=$((total_records + file_records))
        else
            warn "File not found, skipping: $input_file"
        fi
    done
    
    info "Merging completed: $total_records total records"
    echo "$output_file"
}

validate_data() {
    local input_file="$1"
    
    info "Validating data: $input_file"
    
    if [ ! -f "$input_file" ]; then
        error "Input file not found: $input_file"
    fi
    
    python3 << PYTHON_SCRIPT
import json
import sys

input_file = "$input_file"

required_fields = ['content', 'book', 'chapter', 'paragraph_index']
stats = {
    'total': 0,
    'errors': 0,
    'missing_fields': {},
    'by_book': {},
    'avg_length': 0,
    'total_length': 0
}

errors = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        
        stats['total'] += 1
        
        try:
            record = json.loads(line)
            
            for field in required_fields:
                if field not in record:
                    errors.append(f"Line {line_num}: Missing required field '{field}'")
                    stats['errors'] += 1
                    break
            
            content = record.get('content', '')
            stats['total_length'] += len(content)
            
            book = record.get('book', 'unknown')
            stats['by_book'][book] = stats['by_book'].get(book, 0) + 1
            
        except json.JSONDecodeError as e:
            errors.append(f"Line {line_num}: Invalid JSON - {e}")
            stats['errors'] += 1

if stats['total'] > 0:
    stats['avg_length'] = stats['total_length'] / stats['total']

print("\n" + "="*50)
print("VALIDATION REPORT")
print("="*50)
print(f"Total records: {stats['total']}")
print(f"Total errors: {stats['errors']}")
print(f"Average content length: {stats['avg_length']:.1f} characters")
print(f"Error rate: {(stats['errors']/stats['total']*100):.2f}%")
print(f"\nRecords by book:")
for book, count in sorted(stats['by_book'].items(), key=lambda x: -x[1])[:10]:
    print(f"  {book}: {count}")
print("="*50)

if errors:
    print(f"\nFirst 10 errors:")
    for error in errors[:10]:
        print(f"  - {error}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")

if stats['errors'] == 0 and stats['total'] > 0:
    print("\n✓ Validation PASSED")
    sys.exit(0)
else:
    print("\n✗ Validation FAILED")
    sys.exit(1)
PYTHON_SCRIPT
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        info "Data validation passed"
    else
        error "Data validation failed"
    fi
}

tokenize_data() {
    local input_file="$1"
    local output_file="${2:-$TOKENIZED_DIR/tokenized.jsonl}"
    
    info "Tokenizing data: $input_file -> $output_file"
    
    if [ ! -f "$input_file" ]; then
        error "Input file not found: $input_file"
    fi
    
    mkdir -p "$TOKENIZED_DIR"
    
    python3 << PYTHON_SCRIPT
import json
import sys
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from tokenizer import ClassicalTokenizer
    tokenizer = ClassicalTokenizer()
    use_custom_tokenizer = True
except ImportError:
    use_custom_tokenizer = False
    print("Warning: ClassicalTokenizer not found, using char-level tokenization")

input_file = "$input_file"
output_file = "$output_file"

tokenized_count = 0
total_count = 0

with open(input_file, 'r', encoding='utf-8') as fin:
    with open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_count += 1
            try:
                record = json.loads(line.strip())
                content = record.get('content', '')
                
                if use_custom_tokenizer:
                    tokens = tokenizer.encode(content)
                else:
                    tokens = [ord(c) for c in content]
                
                record['tokens'] = tokens
                record['token_count'] = len(tokens)
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                tokenized_count += 1
                
            except Exception as e:
                print(f"Error processing line {total_count}: {e}")
                continue

print(f"Tokenized: {tokenized_count}/{total_count} records")
PYTHON_SCRIPT
    
    info "Tokenization completed: $tokenized_count records"
}

show_stats() {
    info "Dataset statistics"
    
    echo ""
    echo "Directory: $DATA_DIR"
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ File                           │ Records │ Size            │"
    echo "├─────────────────────────────────────────────────────────────┤"
    
    for file in "$PROCESSED_DIR"/*.jsonl "$DATA_DIR"/*.jsonl "$DATA_DIR"/*.json 2>/dev/null; do
        if [ -f "$file" ]; then
            local records
            records=$(wc -l < "$file" 2>/dev/null || echo "0")
            local size
            size=$(du -h "$file" | cut -f1)
            local filename
            filename=$(basename "$file")
            printf "│ %-30s │ %7s │ %-15s │\n" "$filename" "$records" "$size"
        fi
    done
    
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
    
    if [ -f "$PROCESSED_DIR/merged.jsonl" ]; then
        info "Sample from merged dataset:"
        head -n 3 "$PROCESSED_DIR/merged.jsonl" | python3 -m json.tool --no-ensure-ascii 2>/dev/null || head -n 3 "$PROCESSED_DIR/merged.jsonl"
    fi
}

show_help() {
    cat << EOF
$0 - Data Pipeline for Lingmao Moyun

USAGE:
    $0 [OPTIONS] [COMMAND]

COMMANDS:
    download          Download data from sources
    clean             Clean raw data
    convert           Convert data formats
    merge             Merge multiple datasets
    tokenize          Tokenize text data
    validate          Validate data quality
    stats             Show dataset statistics
    full              Run complete pipeline (download → clean → merge)

OPTIONS:
    -s, --source      Specify data source (poetry, classical, terms, all)
    -i, --input       Input file path
    -o, --output      Output file path
    -w, --workers     Number of worker threads (default: 1)
    -f, --force       Force overwrite existing files
    -v, --verbose     Enable verbose output
    -n, --dry-run     Show what would be done without executing
    -h, --help        Show this help message

EXAMPLES:
    $0 --full                          # Run complete pipeline
    $0 --download --source poetry      # Download poetry data
    $0 --clean -i raw/data.jsonl -o processed/clean.jsonl
    $0 --merge -o merged.jsonl          # Merge all datasets
    $0 --validate -i dataset/merged.jsonl
    $0 --stats                          # Show dataset statistics

DATA SOURCES:
    poetry      - Classical poetry (Tang, Song, etc.)
    classical   - Classical texts (Analects, Dao De Jing, etc.)
    terms       - Classical terminology
    all         - All available sources

For more information, see:
    docs/data/README.md        - Data documentation overview
    docs/data/getting_started.md - Data acquisition guide
    docs/data/formats.md       - Data format specifications
    DATASET.md                 - Complete dataset documentation

EOF
}

main() {
    local command=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            download|clean|convert|merge|tokenize|validate|stats|full)
                command="$1"
                shift
                ;;
            -s|--source)
                DOWNLOAD_SOURCES+=("$2")
                shift 2
                ;;
            -i|--input)
                INPUT_FILE="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -w|--workers)
                WORKERS="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                error "Unknown option: $1"
                ;;
            *)
                error "Unknown argument: $1"
                ;;
        esac
    done
    
    if [ -z "$command" ]; then
        show_help
        exit 1
    fi
    
    create_directories
    check_dependencies
    
    case "$command" in
        download)
            if [ ${#DOWNLOAD_SOURCES[@]} -eq 0 ]; then
                DOWNLOAD_SOURCES=("all")
            fi
            for source in "${DOWNLOAD_SOURCES[@]}"; do
                download_data "$source"
            done
            ;;
        clean)
            INPUT_FILE="${INPUT_FILE:-$PROCESSED_DIR/poetry.jsonl}"
            OUTPUT_FILE="${OUTPUT_FILE:-$PROCESSED_DIR/cleaned.jsonl}"
            clean_data "$INPUT_FILE" "$OUTPUT_FILE"
            ;;
        convert)
            INPUT_FILE="${INPUT_FILE:-$PROCESSED_DIR/merged.jsonl}"
            OUTPUT_FILE="${OUTPUT_FILE:-$PROCESSED_DIR/converted.jsonl}"
            convert_format "$INPUT_FILE" "$OUTPUT_FILE"
            ;;
        merge)
            OUTPUT_FILE="${OUTPUT_FILE:-$PROCESSED_DIR/merged.jsonl}"
            merge_datasets "$OUTPUT_FILE"
            ;;
        tokenize)
            INPUT_FILE="${INPUT_FILE:-$PROCESSED_DIR/merged.jsonl}"
            OUTPUT_FILE="${OUTPUT_FILE:-$TOKENIZED_DIR/tokenized.jsonl}"
            tokenize_data "$INPUT_FILE" "$OUTPUT_FILE"
            ;;
        validate)
            INPUT_FILE="${INPUT_FILE:-$PROCESSED_DIR/merged.jsonl}"
            validate_data "$INPUT_FILE"
            ;;
        stats)
            show_stats
            ;;
        full)
            info "Running full data pipeline..."
            
            download_data "all"
            
            if [ -f "$PROCESSED_DIR/poetry.jsonl" ]; then
                clean_data "$PROCESSED_DIR/poetry.jsonl" "$PROCESSED_DIR/poetry_clean.jsonl"
            fi
            
            if [ -f "$DATA_DIR/classical.jsonl" ]; then
                clean_data "$DATA_DIR/classical.jsonl" "$PROCESSED_DIR/classical_clean.jsonl"
            fi
            
            merge_datasets "$PROCESSED_DIR/merged.jsonl" \
                "$PROCESSED_DIR/poetry_clean.jsonl" \
                "$PROCESSED_DIR/classical_clean.jsonl" \
                "$PROCESSED_DIR/terms.jsonl"
            
            validate_data "$PROCESSED_DIR/merged.jsonl"
            
            show_stats
            
            info "Full pipeline completed successfully!"
            info "Output: $PROCESSED_DIR/merged.jsonl"
            ;;
    esac
}

main "$@"
