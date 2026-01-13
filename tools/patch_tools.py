"""
Patch Tools - Tools for incremental code modifications (diff/patch).
"""

import re
from pathlib import Path
from typing import Optional
from difflib import unified_diff, SequenceMatcher

from .base import BaseTool, ToolResult


class ApplyPatchTool(BaseTool):
    """Tool for applying patches/diffs to files."""
    
    @property
    def name(self) -> str:
        return "apply_patch"
    
    @property
    def description(self) -> str:
        return "Apply a unified diff patch to a file. Useful for incremental code modifications."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to patch"
                },
                "patch": {
                    "type": "string",
                    "description": "Unified diff patch content"
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview changes without applying",
                    "default": True
                }
            },
            "required": ["path", "patch"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        patch = kwargs.get("patch")
        dry_run = kwargs.get("dry_run", True)
        
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult.error_result(f"File not found: {path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                original_lines = f.readlines()
            
            # Parse and apply patch
            new_lines, applied_hunks, failed_hunks = self._apply_unified_diff(
                original_lines, patch
            )
            
            if failed_hunks:
                return ToolResult.error_result(
                    f"Failed to apply {len(failed_hunks)} hunk(s). "
                    f"File may have changed since patch was created."
                )
            
            if not applied_hunks:
                return ToolResult.success_result(
                    "No changes to apply (patch may already be applied).",
                    data={"changes": 0}
                )
            
            new_content = "".join(new_lines)
            
            if not dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            
            # Show changes
            diff_output = list(unified_diff(
                original_lines,
                new_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm=""
            ))
            
            action = "Would apply" if dry_run else "Applied"
            output = f"{action} {len(applied_hunks)} hunk(s) to {path}:\n\n"
            output += "\n".join(diff_output[:50])
            
            if len(diff_output) > 50:
                output += f"\n... ({len(diff_output) - 50} more lines)"
            
            if dry_run:
                output += "\n\n⚠️  Dry run - file not modified. Set dry_run=False to apply."
            
            return ToolResult.success_result(
                output,
                data={"hunks_applied": len(applied_hunks), "dry_run": dry_run}
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Patch error: {str(e)}")
    
    def _apply_unified_diff(self, original_lines: list[str], patch: str) -> tuple:
        """
        Apply a unified diff to lines.
        Returns (new_lines, applied_hunks, failed_hunks).
        """
        applied_hunks = []
        failed_hunks = []
        result_lines = original_lines.copy()
        offset = 0
        
        # Parse patch into hunks
        hunks = self._parse_unified_diff(patch)
        
        for hunk in hunks:
            start_line = hunk["start_line"] - 1 + offset
            
            # Try to apply hunk
            success, new_offset = self._apply_hunk(
                result_lines, hunk, start_line
            )
            
            if success:
                applied_hunks.append(hunk)
                offset += new_offset
            else:
                failed_hunks.append(hunk)
        
        return result_lines, applied_hunks, failed_hunks
    
    def _parse_unified_diff(self, patch: str) -> list[dict]:
        """Parse unified diff into hunks."""
        hunks = []
        current_hunk = None
        
        for line in patch.split("\n"):
            # Hunk header: @@ -start,count +start,count @@
            if line.startswith("@@"):
                if current_hunk:
                    hunks.append(current_hunk)
                
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if match:
                    current_hunk = {
                        "start_line": int(match.group(1)),
                        "remove_lines": [],
                        "add_lines": [],
                        "context_before": [],
                        "context_after": [],
                    }
            elif current_hunk is not None:
                if line.startswith("-") and not line.startswith("---"):
                    current_hunk["remove_lines"].append(line[1:])
                elif line.startswith("+") and not line.startswith("+++"):
                    current_hunk["add_lines"].append(line[1:])
                elif line.startswith(" "):
                    if not current_hunk["remove_lines"] and not current_hunk["add_lines"]:
                        current_hunk["context_before"].append(line[1:])
                    else:
                        current_hunk["context_after"].append(line[1:])
        
        if current_hunk:
            hunks.append(current_hunk)
        
        return hunks
    
    def _apply_hunk(self, lines: list[str], hunk: dict, start: int) -> tuple[bool, int]:
        """Apply a single hunk. Returns (success, line_offset)."""
        # Find the location to apply
        search_start = max(0, start - 3)
        search_end = min(len(lines), start + len(hunk["remove_lines"]) + 3)
        
        # Look for matching context
        for try_start in range(search_start, search_end):
            if self._hunk_matches(lines, hunk, try_start):
                # Apply the hunk
                remove_count = len(hunk["remove_lines"]) + len(hunk["context_before"]) + len(hunk["context_after"])
                add_lines = (
                    [l + "\n" for l in hunk["context_before"]] +
                    [l + "\n" for l in hunk["add_lines"]] +
                    [l + "\n" for l in hunk["context_after"]]
                )
                
                lines[try_start:try_start + remove_count] = add_lines
                return True, len(add_lines) - remove_count
        
        return False, 0
    
    def _hunk_matches(self, lines: list[str], hunk: dict, start: int) -> bool:
        """Check if hunk matches at the given position."""
        expected = (
            hunk["context_before"] +
            hunk["remove_lines"] +
            hunk["context_after"]
        )
        
        if start + len(expected) > len(lines):
            return False
        
        for i, expected_line in enumerate(expected):
            actual_line = lines[start + i].rstrip("\n")
            if actual_line != expected_line:
                return False
        
        return True


class EditBlockTool(BaseTool):
    """Tool for editing specific blocks of code in a file."""
    
    @property
    def name(self) -> str:
        return "edit_block"
    
    @property
    def description(self) -> str:
        return (
            "Edit a specific block of code in a file by providing the old content "
            "and new content. More reliable than patches for small changes."
        )
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_content": {
                    "type": "string",
                    "description": "The exact content to be replaced"
                },
                "new_content": {
                    "type": "string",
                    "description": "The new content to replace with"
                },
                "occurrence": {
                    "type": "integer",
                    "description": "Which occurrence to replace (1 = first, -1 = last, 0 = all)",
                    "default": 1
                }
            },
            "required": ["path", "old_content", "new_content"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        old_content = kwargs.get("old_content")
        new_content = kwargs.get("new_content")
        occurrence = kwargs.get("occurrence", 1)
        
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult.error_result(f"File not found: {path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Find occurrences
            count = content.count(old_content)
            
            if count == 0:
                # Try fuzzy matching
                similar_pos = self._find_similar(content, old_content)
                if similar_pos:
                    return ToolResult.error_result(
                        f"Exact match not found. Did you mean this (line {similar_pos})?\n"
                        f"Use read_file to check the exact content."
                    )
                return ToolResult.error_result(
                    "Content not found in file. Please check that old_content matches exactly."
                )
            
            # Perform replacement
            if occurrence == 0:
                # Replace all
                new_file_content = content.replace(old_content, new_content)
                replaced_count = count
            elif occurrence > 0:
                # Replace nth occurrence
                parts = content.split(old_content)
                if occurrence > count:
                    return ToolResult.error_result(
                        f"Occurrence {occurrence} not found (only {count} occurrences exist)."
                    )
                new_file_content = old_content.join(parts[:occurrence]) + new_content + old_content.join(parts[occurrence:])
                replaced_count = 1
            else:
                # Replace from end (-1 = last)
                idx = abs(occurrence)
                if idx > count:
                    return ToolResult.error_result(
                        f"Occurrence {occurrence} not found (only {count} occurrences exist)."
                    )
                parts = content.rsplit(old_content, idx)
                new_file_content = parts[0] + new_content + old_content.join(parts[1:])
                replaced_count = 1
            
            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_file_content)
            
            # Generate diff preview
            old_lines = content.split("\n")
            new_lines = new_file_content.split("\n")
            diff = list(unified_diff(
                old_lines[:30], new_lines[:30],
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm=""
            ))
            
            output = f"✅ Replaced {replaced_count} occurrence(s) in {path}\n\n"
            if diff:
                output += "Changes:\n" + "\n".join(diff[:30])
                if len(diff) > 30:
                    output += "\n... (truncated)"
            
            return ToolResult.success_result(
                output,
                data={"replaced": replaced_count, "path": str(file_path)}
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Edit error: {str(e)}")
    
    def _find_similar(self, content: str, target: str) -> Optional[int]:
        """Find a similar block in content."""
        lines = content.split("\n")
        target_lines = target.split("\n")
        
        best_ratio = 0
        best_line = None
        
        for i in range(len(lines) - len(target_lines) + 1):
            block = "\n".join(lines[i:i + len(target_lines)])
            ratio = SequenceMatcher(None, block, target).ratio()
            
            if ratio > best_ratio and ratio > 0.6:
                best_ratio = ratio
                best_line = i + 1
        
        return best_line


class InsertCodeTool(BaseTool):
    """Tool for inserting code at a specific location."""
    
    @property
    def name(self) -> str:
        return "insert_code"
    
    @property
    def description(self) -> str:
        return "Insert code at a specific line number in a file."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file"
                },
                "line_number": {
                    "type": "integer",
                    "description": "Line number to insert at (1-indexed, inserts before this line)"
                },
                "content": {
                    "type": "string",
                    "description": "Code content to insert"
                },
                "after": {
                    "type": "boolean",
                    "description": "If true, insert after the line instead of before",
                    "default": False
                }
            },
            "required": ["path", "line_number", "content"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        line_number = kwargs.get("line_number")
        content = kwargs.get("content")
        after = kwargs.get("after", False)
        
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult.error_result(f"File not found: {path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Validate line number
            if line_number < 1 or line_number > len(lines) + 1:
                return ToolResult.error_result(
                    f"Line number {line_number} out of range (1-{len(lines) + 1})."
                )
            
            # Ensure content ends with newline
            if not content.endswith("\n"):
                content += "\n"
            
            # Calculate insertion point
            insert_idx = line_number - 1
            if after:
                insert_idx += 1
            
            # Insert content
            content_lines = content.split("\n")
            if content_lines[-1] == "":
                content_lines = content_lines[:-1]
            content_lines = [line + "\n" for line in content_lines]
            
            new_lines = lines[:insert_idx] + content_lines + lines[insert_idx:]
            
            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            
            position = "after" if after else "before"
            return ToolResult.success_result(
                f"✅ Inserted {len(content_lines)} line(s) {position} line {line_number} in {path}",
                data={
                    "lines_inserted": len(content_lines),
                    "path": str(file_path),
                    "at_line": insert_idx + 1
                }
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Insert error: {str(e)}")


class CreateDiffTool(BaseTool):
    """Tool for creating a diff between two versions of content."""
    
    @property
    def name(self) -> str:
        return "create_diff"
    
    @property
    def description(self) -> str:
        return "Create a unified diff between two versions of code."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "original": {
                    "type": "string",
                    "description": "Original content"
                },
                "modified": {
                    "type": "string",
                    "description": "Modified content"
                },
                "filename": {
                    "type": "string",
                    "description": "Filename for the diff header",
                    "default": "file"
                }
            },
            "required": ["original", "modified"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        original = kwargs.get("original")
        modified = kwargs.get("modified")
        filename = kwargs.get("filename", "file")
        
        try:
            original_lines = original.split("\n")
            modified_lines = modified.split("\n")
            
            diff = list(unified_diff(
                original_lines,
                modified_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                lineterm=""
            ))
            
            if not diff:
                return ToolResult.success_result(
                    "No differences found.",
                    data={"has_changes": False}
                )
            
            diff_content = "\n".join(diff)
            
            # Count changes
            additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
            
            return ToolResult.success_result(
                f"Diff (+{additions}/-{deletions} lines):\n\n{diff_content}",
                data={
                    "has_changes": True,
                    "additions": additions,
                    "deletions": deletions,
                    "diff": diff_content
                }
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Diff error: {str(e)}")
