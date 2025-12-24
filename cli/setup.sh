#!/bin/bash
# Setup script for Zunigram CLI and E2E tests

set -e  # Exit on error

echo "======================================"
echo "Zunigram Setup Script"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Step 1: Check Python version
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Step 2: Check Rust/Cargo
print_status "Checking Rust toolchain..."
if command -v cargo &> /dev/null; then
    CARGO_VERSION=$(cargo --version | cut -d' ' -f2)
    print_success "Cargo $CARGO_VERSION found"
else
    print_error "Cargo not found. Please install Rust: https://rustup.rs/"
    exit 1
fi

# Step 3: Create virtual environment (optional but recommended)
if [ ! -d ".venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

print_status "Activating virtual environment..."
source .venv/bin/activate || {
    print_error "Failed to activate virtual environment"
    exit 1
}

# Step 4: Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "llm/requirements.txt" ]; then
    pip install --upgrade pip > /dev/null
    pip install -r llm/requirements.txt
    print_success "Core dependencies installed"
else
    print_warning "llm/requirements.txt not found"
fi

if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
    print_success "Development dependencies installed"
fi

# Step 5: Build zunigram-py Python bindings
print_status "Building zunigram-py Python bindings..."
if [ -d "crates/zunigram-py" ]; then
    # Check if maturin is installed
    if ! command -v maturin &> /dev/null; then
        print_status "Installing maturin..."
        pip install maturin
    fi
    
    cd crates/zunigram-py
    maturin develop --release
    cd "$PROJECT_ROOT"
    print_success "zunigram-py bindings built successfully"
else
    print_error "crates/zunigram-py directory not found"
    exit 1
fi

# Step 6: Build Rust prover binary
print_status "Building Rust prover binary..."
print_status "(This may take a few minutes...)"
cargo build -p prover --release --example prove_from_json

if [ -f "target/release/examples/prove_from_json" ]; then
    print_success "Prover binary built successfully"
else
    print_error "Failed to build prover binary"
    exit 1
fi

# Step 7: Verify installation
print_status "Verifying installation..."

# Test zunigram_py import
python3 -c "import zunigram_py; print('zunigram_py version:', zunigram_py.__doc__)" && \
    print_success "zunigram_py bindings working" || \
    print_error "zunigram_py import failed"

# Test CLI import
python3 -c "import sys; sys.path.insert(0, '.'); from llm.watermark import WatermarkConfig" && \
    print_success "CLI modules accessible" || \
    print_error "CLI import failed"

# Check prover binary
if [ -x "target/release/examples/prove_from_json" ]; then
    print_success "Prover binary is executable"
else
    print_warning "Prover binary may not be executable"
fi

# Step 8: Run quick smoke test
print_status "Running smoke test..."
python3 -c "
import zunigram_py

# Test basic functionality
secret_key = zunigram_py.generate_secret_key()
assert len(secret_key) == 8, 'Secret key should have 8 elements'

tokens = [1, 2, 3, 4, 5]
green_count = zunigram_py.count_green(secret_key, tokens)
assert green_count >= 0, 'Green count should be non-negative'

z_score = zunigram_py.compute_z_score(green_count, len(tokens), 0.5)
assert isinstance(z_score, float), 'Z-score should be float'

print('✓ All basic tests passed')
" && print_success "Smoke test passed" || print_error "Smoke test failed"

# Summary
echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment (if not already active):"
echo "   ${GREEN}source .venv/bin/activate${NC}"
echo ""
echo "2. Try the CLI commands:"
echo "   ${GREEN}python cli/zunigram_cli.py generate \"Once upon a time\" -o output.json${NC}"
echo "   ${GREEN}python cli/zunigram_cli.py detect output.json${NC}"
echo "   ${GREEN}python cli/zunigram_cli.py prove output.json -o proof.json${NC}"
echo "   ${GREEN}python cli/zunigram_cli.py verify proof.json${NC}"
echo ""
echo "3. Run the test suite:"
echo "   ${GREEN}python -m pytest tests/ -v${NC}"
echo "   ${GREEN}# or${NC}"
echo "   ${GREEN}python tests/test_e2e.py${NC}"
echo ""
echo "4. Read the documentation:"
echo "   - CLI: ${BLUE}cli/README.md${NC}"
echo "   - Tests: ${BLUE}tests/README.md${NC}"
echo ""

print_success "Setup completed successfully!"

