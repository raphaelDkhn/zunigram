import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON from file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """Save data as JSON to file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation (default 2)
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_secret_key(file_path: str) -> List[int]:
    """Load secret key from JSON file.
    
    Args:
        file_path: Path to secret key file
        
    Returns:
        Secret key as list of 8 u32 values
        
    Raises:
        ValueError: If secret key format is invalid
    """
    data = load_json(file_path)
    
    if "secret_key" in data:
        key = data["secret_key"]
    elif isinstance(data, list):
        key = data
    else:
        raise ValueError("Invalid secret key format. Expected 'secret_key' field or array.")
    
    if not isinstance(key, list) or len(key) != 8:
        raise ValueError(f"Secret key must be a list of 8 integers, got {len(key) if isinstance(key, list) else 'non-list'}")
    
    return key


def save_secret_key(secret_key: List[int], file_path: str) -> None:
    """Save secret key to JSON file.
    
    Args:
        secret_key: List of 8 u32 values
        file_path: Output file path
    """
    save_json({"secret_key": secret_key}, file_path)


def print_section(title: str, color: str = "blue") -> None:
    """Print a section header.
    
    Args:
        title: Section title
        color: Color name (blue, green, red, yellow)
    """
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "reset": "\033[0m"
    }
    
    color_code = colors.get(color, colors["blue"])
    reset = colors["reset"]
    
    print(f"\n{color_code}{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}{reset}\n")


def print_result(label: str, value: Any, indent: int = 0) -> None:
    """Print a labeled result.
    
    Args:
        label: Result label
        value: Result value
        indent: Indentation level
    """
    prefix = "  " * indent
    print(f"{prefix}{label}: {value}")


def print_error(message: str) -> None:
    """Print an error message to stderr.
    
    Args:
        message: Error message
    """
    print(f"\033[91mError: {message}\033[0m", file=sys.stderr)


def print_success(message: str) -> None:
    """Print a success message.
    
    Args:
        message: Success message
    """
    print(f"\033[92m✓ {message}\033[0m")


def print_warning(message: str) -> None:
    """Print a warning message.
    
    Args:
        message: Warning message
    """
    print(f"\033[93m⚠ {message}\033[0m")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def format_percentage(value: float) -> str:
    """Format value as percentage.
    
    Args:
        value: Value between 0 and 1
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

