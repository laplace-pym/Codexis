"""
Search Tools - Advanced code search capabilities (grep-like).
"""

import re
import fnmatch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .base import BaseTool, ToolResult


@dataclass
class SearchMatch:
    """A single search match."""
    file_path: str
    line_number: int
    line_content: str
    match_start: int
    match_end: int


class GrepTool(BaseTool):
    """Tool for searching code using regular expressions (grep-like)."""
    
    @property
    def name(self) -> str:
        return "grep"
    
    @property
    def description(self) -> str:
        return "Search for patterns in files using regular expressions. Similar to grep command."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path to search in"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for file names (e.g., '*.py')",
                    "default": "*"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case insensitive search",
                    "default": False
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before/after match",
                    "default": 0
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 50
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories",
                    "default": True
                }
            },
            "required": ["pattern", "path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        pattern = kwargs.get("pattern")
        path = kwargs.get("path")
        file_pattern = kwargs.get("file_pattern", "*")
        case_insensitive = kwargs.get("case_insensitive", False)
        context_lines = kwargs.get("context_lines", 0)
        max_results = kwargs.get("max_results", 50)
        recursive = kwargs.get("recursive", True)
        
        try:
            # Compile regex
            flags = re.IGNORECASE if case_insensitive else 0
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return ToolResult.error_result(f"Invalid regex pattern: {e}")
            
            search_path = Path(path).expanduser().resolve()
            
            if not search_path.exists():
                return ToolResult.error_result(f"Path not found: {path}")
            
            matches = []
            files_searched = 0
            
            # Get files to search
            if search_path.is_file():
                files = [search_path]
            elif recursive:
                files = list(search_path.rglob(file_pattern))
            else:
                files = list(search_path.glob(file_pattern))
            
            for file_path in files:
                if not file_path.is_file():
                    continue
                
                # Skip binary files
                if self._is_binary(file_path):
                    continue
                
                files_searched += 1
                
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        if regex.search(line):
                            # Get context lines
                            start = max(0, line_num - 1 - context_lines)
                            end = min(len(lines), line_num + context_lines)
                            
                            context = []
                            for i in range(start, end):
                                prefix = ">" if i == line_num - 1 else " "
                                context.append(f"{prefix}{i+1:4d}| {lines[i].rstrip()}")
                            
                            rel_path = file_path.relative_to(search_path) if search_path.is_dir() else file_path.name
                            matches.append({
                                "file": str(rel_path),
                                "line": line_num,
                                "content": "\n".join(context) if context_lines > 0 else line.rstrip(),
                            })
                            
                            if len(matches) >= max_results:
                                break
                    
                except (UnicodeDecodeError, PermissionError):
                    continue
                
                if len(matches) >= max_results:
                    break
            
            if not matches:
                return ToolResult.success_result(
                    f"No matches found for '{pattern}' in {files_searched} files.",
                    data={"matches": 0, "files_searched": files_searched}
                )
            
            # Format output
            output_lines = [f"Found {len(matches)} matches in {files_searched} files:"]
            
            current_file = None
            for match in matches:
                if match["file"] != current_file:
                    current_file = match["file"]
                    output_lines.append(f"\n📄 {current_file}:")
                
                if context_lines > 0:
                    output_lines.append(match["content"])
                    output_lines.append("---")
                else:
                    output_lines.append(f"  {match['line']:4d}: {match['content']}")
            
            if len(matches) >= max_results:
                output_lines.append(f"\n... (limited to {max_results} results)")
            
            return ToolResult.success_result(
                "\n".join(output_lines),
                data={"matches": len(matches), "files_searched": files_searched}
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Search error: {str(e)}")
    
    def _is_binary(self, path: Path) -> bool:
        """Check if file is binary."""
        binary_extensions = {
            ".pyc", ".pyo", ".so", ".o", ".a", ".lib", ".dll", ".exe",
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf",
            ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
            ".mp3", ".mp4", ".avi", ".mov", ".mkv",
            ".ttf", ".otf", ".woff", ".woff2",
        }
        return path.suffix.lower() in binary_extensions


class FindSymbolTool(BaseTool):
    """Tool for finding symbol definitions (functions, classes, variables)."""
    
    @property
    def name(self) -> str:
        return "find_symbol"
    
    @property
    def description(self) -> str:
        return "Find definitions of functions, classes, or variables in code."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol name to find"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in"
                },
                "symbol_type": {
                    "type": "string",
                    "description": "Type of symbol: function, class, variable, all",
                    "enum": ["function", "class", "variable", "all"],
                    "default": "all"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (auto-detected if not specified)",
                    "enum": ["python", "javascript", "typescript", "auto"],
                    "default": "auto"
                }
            },
            "required": ["symbol", "path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        symbol = kwargs.get("symbol")
        path = kwargs.get("path")
        symbol_type = kwargs.get("symbol_type", "all")
        language = kwargs.get("language", "auto")
        
        try:
            search_path = Path(path).expanduser().resolve()
            
            if not search_path.exists():
                return ToolResult.error_result(f"Path not found: {path}")
            
            # Build patterns based on symbol type
            patterns = self._get_patterns(symbol, symbol_type, language)
            
            # Get files based on language
            file_patterns = self._get_file_patterns(language)
            
            matches = []
            
            for file_pattern in file_patterns:
                if search_path.is_file():
                    files = [search_path]
                else:
                    files = list(search_path.rglob(file_pattern))
                
                for file_path in files:
                    if not file_path.is_file():
                        continue
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        
                        for line_num, line in enumerate(lines, 1):
                            for pattern_type, pattern in patterns.items():
                                if pattern.search(line):
                                    rel_path = file_path.relative_to(search_path) if search_path.is_dir() else file_path.name
                                    matches.append({
                                        "file": str(rel_path),
                                        "line": line_num,
                                        "type": pattern_type,
                                        "content": line.strip(),
                                    })
                                    break
                    except (UnicodeDecodeError, PermissionError):
                        continue
            
            if not matches:
                return ToolResult.success_result(
                    f"Symbol '{symbol}' not found.",
                    data={"found": False}
                )
            
            # Format output
            output_lines = [f"Found {len(matches)} definitions of '{symbol}':"]
            
            for match in matches:
                output_lines.append(f"\n📍 {match['file']}:{match['line']} [{match['type']}]")
                output_lines.append(f"   {match['content']}")
            
            return ToolResult.success_result(
                "\n".join(output_lines),
                data={"found": True, "matches": matches}
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Search error: {str(e)}")
    
    def _get_patterns(self, symbol: str, symbol_type: str, language: str) -> dict:
        """Get regex patterns for finding symbols."""
        patterns = {}
        escaped = re.escape(symbol)
        
        if symbol_type in ("function", "all"):
            # Python: def symbol(
            patterns["function"] = re.compile(rf"^\s*(async\s+)?def\s+{escaped}\s*\(")
            # JavaScript/TypeScript: function symbol( or symbol: function(
            if language in ("javascript", "typescript", "auto"):
                patterns["js_function"] = re.compile(rf"(function\s+{escaped}|{escaped}\s*[=:]\s*(async\s+)?function|\s+{escaped}\s*\()")
        
        if symbol_type in ("class", "all"):
            # Python: class Symbol
            patterns["class"] = re.compile(rf"^\s*class\s+{escaped}\s*[:\(]")
            # JavaScript/TypeScript: class Symbol
            if language in ("javascript", "typescript", "auto"):
                patterns["js_class"] = re.compile(rf"class\s+{escaped}\s*[\{{<]")
        
        if symbol_type in ("variable", "all"):
            # Python: symbol = 
            patterns["variable"] = re.compile(rf"^\s*{escaped}\s*=\s*[^=]")
            # JavaScript/TypeScript: const/let/var symbol =
            if language in ("javascript", "typescript", "auto"):
                patterns["js_variable"] = re.compile(rf"(const|let|var)\s+{escaped}\s*[=:]")
        
        return patterns
    
    def _get_file_patterns(self, language: str) -> list[str]:
        """Get file patterns based on language."""
        patterns = {
            "python": ["*.py"],
            "javascript": ["*.js", "*.jsx"],
            "typescript": ["*.ts", "*.tsx"],
            "auto": ["*.py", "*.js", "*.jsx", "*.ts", "*.tsx"],
        }
        return patterns.get(language, patterns["auto"])


class ReplaceInFilesTool(BaseTool):
    """Tool for search and replace across multiple files."""
    
    @property
    def name(self) -> str:
        return "replace_in_files"
    
    @property
    def description(self) -> str:
        return "Search and replace text across multiple files. Supports regex."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "Text or regex pattern to search for"
                },
                "replace": {
                    "type": "string",
                    "description": "Replacement text"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for file names (e.g., '*.py')",
                    "default": "*"
                },
                "use_regex": {
                    "type": "boolean",
                    "description": "Treat search as regex pattern",
                    "default": False
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview changes without making them",
                    "default": True
                }
            },
            "required": ["search", "replace", "path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        search = kwargs.get("search")
        replace = kwargs.get("replace")
        path = kwargs.get("path")
        file_pattern = kwargs.get("file_pattern", "*")
        use_regex = kwargs.get("use_regex", False)
        dry_run = kwargs.get("dry_run", True)
        
        try:
            search_path = Path(path).expanduser().resolve()
            
            if not search_path.exists():
                return ToolResult.error_result(f"Path not found: {path}")
            
            if use_regex:
                try:
                    pattern = re.compile(search)
                except re.error as e:
                    return ToolResult.error_result(f"Invalid regex: {e}")
            else:
                pattern = re.compile(re.escape(search))
            
            changes = []
            
            if search_path.is_file():
                files = [search_path]
            else:
                files = list(search_path.rglob(file_pattern))
            
            for file_path in files:
                if not file_path.is_file():
                    continue
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        original = f.read()
                    
                    # Find all matches
                    matches = list(pattern.finditer(original))
                    if not matches:
                        continue
                    
                    # Perform replacement
                    new_content = pattern.sub(replace, original)
                    
                    rel_path = file_path.relative_to(search_path) if search_path.is_dir() else file_path.name
                    changes.append({
                        "file": str(rel_path),
                        "matches": len(matches),
                        "preview": self._get_preview(original, matches, replace, pattern),
                    })
                    
                    # Actually write if not dry run
                    if not dry_run:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                    
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            if not changes:
                return ToolResult.success_result(
                    f"No matches found for '{search}'.",
                    data={"changes": 0}
                )
            
            # Format output
            total_matches = sum(c["matches"] for c in changes)
            action = "Would change" if dry_run else "Changed"
            
            output_lines = [
                f"{action} {total_matches} occurrences in {len(changes)} files:"
            ]
            
            for change in changes:
                output_lines.append(f"\n📄 {change['file']} ({change['matches']} matches):")
                output_lines.append(change["preview"])
            
            if dry_run:
                output_lines.append("\n⚠️  Dry run - no files modified. Set dry_run=False to apply changes.")
            
            return ToolResult.success_result(
                "\n".join(output_lines),
                data={"changes": len(changes), "total_matches": total_matches, "dry_run": dry_run}
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Replace error: {str(e)}")
    
    def _get_preview(self, content: str, matches: list, replace: str, pattern: re.Pattern) -> str:
        """Get a preview of changes."""
        lines = content.split("\n")
        previews = []
        shown_lines = set()
        
        for match in matches[:3]:  # Show first 3 matches
            line_num = content[:match.start()].count("\n")
            if line_num in shown_lines:
                continue
            shown_lines.add(line_num)
            
            original_line = lines[line_num]
            new_line = pattern.sub(replace, original_line)
            
            previews.append(f"  - {original_line.strip()}")
            previews.append(f"  + {new_line.strip()}")
        
        if len(matches) > 3:
            previews.append(f"  ... and {len(matches) - 3} more")
        
        return "\n".join(previews)
