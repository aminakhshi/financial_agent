from pathlib import Path
from typing import List


def load_symbol_file(path: Path) -> List[str]:
    if not path.exists():
        return []

    seen = set()
    symbols: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        symbol = raw_line.strip().upper()
        if not symbol or symbol.startswith("#"):
            continue
        normalized = symbol.replace(".", "-")
        if normalized in seen:
            continue
        seen.add(normalized)
        symbols.append(normalized)
    return symbols


def load_sp500_symbols(base_dir: Path) -> List[str]:
    return load_symbol_file(base_dir / "src" / "data" / "static" / "sp500_symbols.txt")
