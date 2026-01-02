#!/bin/bash
#
# Embedding Atlas - Qualitative Coding Tool
# Install and run script
#
# Usage:
#   ./run.sh              # Install dependencies and start the app
#   ./run.sh --dev        # Start in development mode (with hot reload)
#   ./run.sh --install    # Only install dependencies
#   ./run.sh --help       # Show help
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║         Embedding Atlas - Qualitative Coding Tool         ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

show_help() {
    print_header
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --dev       Start in development mode with hot reload"
    echo "  --install   Only install dependencies (don't start the app)"
    echo "  --build     Build for production"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh              # Install and start the app"
    echo "  ./run.sh --dev        # Start with hot reload for development"
    echo "  ./run.sh --install    # Just install dependencies"
    echo ""
}

check_requirements() {
    print_info "Checking requirements..."

    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed"
        echo "Please install Node.js 18+ from https://nodejs.org/"
        exit 1
    fi
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        print_error "Node.js 18+ is required (found v$NODE_VERSION)"
        exit 1
    fi
    print_success "Node.js $(node -v)"

    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed"
        exit 1
    fi
    print_success "npm $(npm -v)"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        echo "Please install Python 3.11+ from https://python.org/"
        exit 1
    fi
    print_success "Python $(python3 --version | cut -d' ' -f2)"

    # Check for uv (Python package manager)
    if ! command -v uv &> /dev/null; then
        print_warning "uv is not installed, installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    print_success "uv installed"

    # Check for Rust (needed for WASM)
    if ! command -v rustc &> /dev/null; then
        print_warning "Rust is not installed"
        echo "Some features require Rust. Install from https://rustup.rs/"
        echo "Continuing without WASM support..."
    else
        print_success "Rust $(rustc --version | cut -d' ' -f2)"
    fi
}

install_dependencies() {
    print_info "Installing dependencies..."

    # Install Node.js dependencies
    if [ ! -d "node_modules" ]; then
        print_info "Installing npm packages..."
        npm install
    else
        print_success "npm packages already installed"
    fi

    # Install Python dependencies for backend
    if [ -d "packages/backend" ]; then
        print_info "Installing Python packages..."
        cd packages/backend
        uv sync --frozen 2>/dev/null || uv sync
        cd "$SCRIPT_DIR"
    fi

    print_success "Dependencies installed"
}

build_wasm() {
    if command -v rustc &> /dev/null && [ -f "scripts/build_wasm.sh" ]; then
        print_info "Building WASM modules..."

        # Check if wasm32 target is installed
        if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
            rustup target add wasm32-unknown-unknown
        fi

        # Check if wasm-bindgen-cli is installed
        if ! command -v wasm-bindgen &> /dev/null; then
            cargo install wasm-bindgen-cli
        fi

        ./scripts/build_wasm.sh
        print_success "WASM modules built"
    else
        print_warning "Skipping WASM build (Rust not available)"
    fi
}

build_app() {
    print_info "Building application..."
    npm run build
    print_success "Build complete"
}

start_dev() {
    print_info "Starting development server..."
    echo ""
    echo -e "${GREEN}The app will open at: ${YELLOW}http://localhost:5173${NC}"
    echo -e "${BLUE}Press Ctrl+C to stop${NC}"
    echo ""

    cd packages/viewer
    npm run dev
}

start_app() {
    print_info "Starting application..."

    # First build if needed
    if [ ! -d "packages/viewer/dist" ]; then
        build_app
    fi

    echo ""
    echo -e "${GREEN}The app will open at: ${YELLOW}http://localhost:4173${NC}"
    echo -e "${BLUE}Press Ctrl+C to stop${NC}"
    echo ""

    cd packages/viewer
    npm run preview
}

# Parse arguments
MODE="run"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            MODE="dev"
            shift
            ;;
        --install)
            MODE="install"
            shift
            ;;
        --build)
            MODE="build"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
print_header
check_requirements
install_dependencies

case $MODE in
    install)
        print_success "Installation complete!"
        echo ""
        echo "Run ./run.sh to start the application"
        ;;
    dev)
        start_dev
        ;;
    build)
        build_wasm
        build_app
        print_success "Build complete!"
        echo ""
        echo "Run ./run.sh to start the application"
        ;;
    run)
        start_app
        ;;
esac
