"""
Analysis Tools - Code analysis, test generation, and summarization.
"""

import re
import ast
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .base import BaseTool, ToolResult


@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    args: list[str]
    docstring: Optional[str]
    line_number: int
    return_type: Optional[str] = None


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    methods: list[FunctionInfo]
    docstring: Optional[str]
    line_number: int


class AnalyzeCodeTool(BaseTool):
    """Tool for analyzing Python code structure."""
    
    @property
    def name(self) -> str:
        return "analyze_code"
    
    @property
    def description(self) -> str:
        return "Analyze Python code structure: find functions, classes, and their signatures."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Python file to analyze"
                },
                "include_private": {
                    "type": "boolean",
                    "description": "Include private methods/functions (starting with _)",
                    "default": False
                }
            },
            "required": ["path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        include_private = kwargs.get("include_private", False)
        
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult.error_result(f"File not found: {path}")
            
            if file_path.suffix != ".py":
                return ToolResult.error_result("Only Python files are supported")
            
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                return ToolResult.error_result(f"Syntax error in file: {e}")
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    if not include_private and node.name.startswith("_") and not node.name.startswith("__"):
                        continue
                    
                    # Skip methods (will be captured in classes)
                    parent_is_class = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef):
                            for child in ast.iter_child_nodes(parent):
                                if child is node:
                                    parent_is_class = True
                                    break
                    
                    if not parent_is_class:
                        functions.append(self._extract_function_info(node))
                
                elif isinstance(node, ast.ClassDef):
                    if not include_private and node.name.startswith("_"):
                        continue
                    classes.append(self._extract_class_info(node, include_private))
            
            # Format output
            output_lines = [f"📄 Analysis of {file_path.name}:\n"]
            
            if imports:
                output_lines.append("📦 Imports:")
                for imp in imports[:10]:
                    output_lines.append(f"  - {imp}")
                if len(imports) > 10:
                    output_lines.append(f"  ... and {len(imports) - 10} more")
                output_lines.append("")
            
            if classes:
                output_lines.append(f"🏛️  Classes ({len(classes)}):")
                for cls in classes:
                    output_lines.append(f"\n  class {cls.name} (line {cls.line_number}):")
                    if cls.docstring:
                        doc_preview = cls.docstring.split("\n")[0][:50]
                        output_lines.append(f"    \"{doc_preview}...\"")
                    for method in cls.methods:
                        args_str = ", ".join(method.args[:5])
                        if len(method.args) > 5:
                            args_str += ", ..."
                        output_lines.append(f"    • {method.name}({args_str})")
                output_lines.append("")
            
            if functions:
                output_lines.append(f"🔧 Functions ({len(functions)}):")
                for func in functions:
                    args_str = ", ".join(func.args[:5])
                    if len(func.args) > 5:
                        args_str += ", ..."
                    ret = f" -> {func.return_type}" if func.return_type else ""
                    output_lines.append(f"  • {func.name}({args_str}){ret} (line {func.line_number})")
                    if func.docstring:
                        doc_preview = func.docstring.split("\n")[0][:40]
                        output_lines.append(f"      \"{doc_preview}...\"")
            
            return ToolResult.success_result(
                "\n".join(output_lines),
                data={
                    "classes": len(classes),
                    "functions": len(functions),
                    "imports": len(imports),
                }
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Analysis error: {str(e)}")
    
    def _extract_function_info(self, node) -> FunctionInfo:
        """Extract function information from AST node."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        
        docstring = ast.get_docstring(node)
        
        return_type = None
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except:
                pass
        
        return FunctionInfo(
            name=node.name,
            args=args,
            docstring=docstring,
            line_number=node.lineno,
            return_type=return_type,
        )
    
    def _extract_class_info(self, node: ast.ClassDef, include_private: bool) -> ClassInfo:
        """Extract class information from AST node."""
        methods = []
        
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not include_private and child.name.startswith("_") and not child.name.startswith("__"):
                    continue
                methods.append(self._extract_function_info(child))
        
        return ClassInfo(
            name=node.name,
            methods=methods,
            docstring=ast.get_docstring(node),
            line_number=node.lineno,
        )


class GenerateTestsTool(BaseTool):
    """Tool for generating test cases for Python code."""
    
    @property
    def name(self) -> str:
        return "generate_tests"
    
    @property
    def description(self) -> str:
        return "Generate pytest test cases for Python functions/classes."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Python file to generate tests for"
                },
                "function_name": {
                    "type": "string",
                    "description": "Specific function to test (optional, tests all if not provided)"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to write test file (optional)"
                },
                "test_style": {
                    "type": "string",
                    "description": "Test style: pytest, unittest",
                    "enum": ["pytest", "unittest"],
                    "default": "pytest"
                }
            },
            "required": ["path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        function_name = kwargs.get("function_name")
        output_path = kwargs.get("output_path")
        test_style = kwargs.get("test_style", "pytest")
        
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult.error_result(f"File not found: {path}")
            
            if file_path.suffix != ".py":
                return ToolResult.error_result("Only Python files are supported")
            
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                return ToolResult.error_result(f"Syntax error: {e}")
            
            # Extract functions to test
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_"):
                        continue
                    if function_name and node.name != function_name:
                        continue
                    functions.append(self._extract_function_info(node))
            
            if not functions:
                return ToolResult.error_result(
                    f"No testable functions found" + 
                    (f" matching '{function_name}'" if function_name else "")
                )
            
            # Generate test code
            module_name = file_path.stem
            test_code = self._generate_test_code(module_name, functions, test_style)
            
            # Write to file if path provided
            if output_path:
                out_path = Path(output_path).expanduser().resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(test_code)
                return ToolResult.success_result(
                    f"✅ Generated tests for {len(functions)} function(s)\n"
                    f"Written to: {out_path}\n\n{test_code}",
                    data={"functions_tested": len(functions), "output_path": str(out_path)}
                )
            
            return ToolResult.success_result(
                f"Generated tests for {len(functions)} function(s):\n\n{test_code}",
                data={"functions_tested": len(functions), "test_code": test_code}
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Test generation error: {str(e)}")
    
    def _extract_function_info(self, node) -> FunctionInfo:
        """Extract function information from AST node."""
        args = []
        for arg in node.args.args:
            if arg.arg != "self":
                args.append(arg.arg)
        
        return FunctionInfo(
            name=node.name,
            args=args,
            docstring=ast.get_docstring(node),
            line_number=node.lineno,
        )
    
    def _generate_test_code(self, module_name: str, functions: list[FunctionInfo], style: str) -> str:
        """Generate test code for functions."""
        lines = []
        
        if style == "pytest":
            lines.extend([
                f'"""',
                f'Tests for {module_name}',
                f'',
                f'Auto-generated test file. Customize test cases as needed.',
                f'"""',
                f'',
                f'import pytest',
                f'from {module_name} import *',
                f'',
                f'',
            ])
            
            for func in functions:
                lines.extend([
                    f'class Test{self._to_pascal_case(func.name)}:',
                    f'    """Tests for {func.name}"""',
                    f'',
                    f'    def test_{func.name}_basic(self):',
                    f'        """Test basic functionality of {func.name}"""',
                ])
                
                # Generate basic test based on args
                if func.args:
                    args_str = ", ".join([self._get_default_value(arg) for arg in func.args])
                    lines.append(f'        result = {func.name}({args_str})')
                    lines.append(f'        assert result is not None  # TODO: Add specific assertions')
                else:
                    lines.append(f'        result = {func.name}()')
                    lines.append(f'        assert result is not None  # TODO: Add specific assertions')
                
                lines.extend([
                    f'',
                    f'    def test_{func.name}_edge_cases(self):',
                    f'        """Test edge cases for {func.name}"""',
                    f'        # TODO: Add edge case tests',
                    f'        pass',
                    f'',
                    f'    def test_{func.name}_error_handling(self):',
                    f'        """Test error handling for {func.name}"""',
                    f'        # TODO: Add error handling tests',
                ])
                
                if func.args:
                    lines.append(f'        with pytest.raises(Exception):')
                    lines.append(f'            {func.name}(None)  # Example: test with invalid input')
                else:
                    lines.append(f'        pass')
                
                lines.extend(['', ''])
        
        else:  # unittest
            lines.extend([
                f'"""',
                f'Tests for {module_name}',
                f'"""',
                f'',
                f'import unittest',
                f'from {module_name} import *',
                f'',
                f'',
            ])
            
            for func in functions:
                lines.extend([
                    f'class Test{self._to_pascal_case(func.name)}(unittest.TestCase):',
                    f'    """Tests for {func.name}"""',
                    f'',
                    f'    def test_{func.name}_basic(self):',
                    f'        """Test basic functionality"""',
                ])
                
                if func.args:
                    args_str = ", ".join([self._get_default_value(arg) for arg in func.args])
                    lines.append(f'        result = {func.name}({args_str})')
                    lines.append(f'        self.assertIsNotNone(result)')
                else:
                    lines.append(f'        result = {func.name}()')
                    lines.append(f'        self.assertIsNotNone(result)')
                
                lines.extend([
                    f'',
                    f'',
                ])
            
            lines.extend([
                f"if __name__ == '__main__':",
                f'    unittest.main()',
            ])
        
        return "\n".join(lines)
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert snake_case to PascalCase."""
        return "".join(word.capitalize() for word in name.split("_"))
    
    def _get_default_value(self, arg_name: str) -> str:
        """Get a sensible default test value based on argument name."""
        arg_lower = arg_name.lower()
        
        if any(x in arg_lower for x in ["num", "count", "size", "length", "index", "n", "id"]):
            return "1"
        if any(x in arg_lower for x in ["name", "text", "string", "str", "msg", "message"]):
            return '"test"'
        if any(x in arg_lower for x in ["list", "items", "array"]):
            return "[]"
        if any(x in arg_lower for x in ["dict", "map", "data", "config"]):
            return "{}"
        if any(x in arg_lower for x in ["flag", "is_", "has_", "enable", "disable", "bool"]):
            return "True"
        if any(x in arg_lower for x in ["path", "file", "dir"]):
            return '"/tmp/test"'
        
        return "None"


class SummarizeTool(BaseTool):
    """Tool for summarizing file contents."""
    
    @property
    def name(self) -> str:
        return "summarize"
    
    @property
    def description(self) -> str:
        return "Generate a summary of a file's contents (code or text)."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file to summarize"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum summary length in characters",
                    "default": 500
                }
            },
            "required": ["path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        max_length = kwargs.get("max_length", 500)
        
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult.error_result(f"File not found: {path}")
            
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            file_ext = file_path.suffix.lower()
            
            # Generate summary based on file type
            if file_ext == ".py":
                summary = self._summarize_python(content, file_path.name)
            elif file_ext in (".js", ".ts", ".jsx", ".tsx"):
                summary = self._summarize_javascript(content, file_path.name)
            elif file_ext in (".md", ".markdown"):
                summary = self._summarize_markdown(content, file_path.name)
            else:
                summary = self._summarize_text(content, file_path.name)
            
            # Truncate if needed
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return ToolResult.success_result(
                summary,
                data={
                    "file": str(file_path),
                    "size": len(content),
                    "lines": content.count("\n") + 1,
                }
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Summary error: {str(e)}")
    
    def _summarize_python(self, content: str, filename: str) -> str:
        """Summarize Python file."""
        lines = []
        
        try:
            tree = ast.parse(content)
            
            # Get module docstring
            docstring = ast.get_docstring(tree)
            if docstring:
                lines.append(f"📄 {filename}")
                lines.append(f"   {docstring.split(chr(10))[0]}")
                lines.append("")
            else:
                lines.append(f"📄 {filename}")
                lines.append("")
            
            # Count elements
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                        and not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree) if n in ast.walk(p))]
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            
            stats = []
            if classes:
                stats.append(f"{len(classes)} class(es)")
            if functions:
                stats.append(f"{len(functions)} function(s)")
            if imports:
                stats.append(f"{len(imports)} import(s)")
            
            if stats:
                lines.append(f"📊 Contains: {', '.join(stats)}")
            
            # List main components
            if classes:
                lines.append("\n🏛️  Classes:")
                for cls in classes[:5]:
                    lines.append(f"   • {cls.name}")
                if len(classes) > 5:
                    lines.append(f"   ... and {len(classes) - 5} more")
            
            if functions:
                top_level_funcs = [f for f in functions if not f.name.startswith("_")][:5]
                if top_level_funcs:
                    lines.append("\n🔧 Functions:")
                    for func in top_level_funcs:
                        lines.append(f"   • {func.name}()")
            
        except SyntaxError:
            lines.append(f"📄 {filename}")
            lines.append(f"   ({content.count(chr(10)) + 1} lines of Python code)")
            lines.append("   ⚠️  Contains syntax errors")
        
        return "\n".join(lines)
    
    def _summarize_javascript(self, content: str, filename: str) -> str:
        """Summarize JavaScript/TypeScript file."""
        lines = [f"📄 {filename}"]
        
        # Simple regex-based analysis
        exports = re.findall(r"export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)", content)
        functions = re.findall(r"(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s*)?\(|[=:]\s*function|\()", content)
        classes = re.findall(r"class\s+(\w+)", content)
        imports = re.findall(r"import\s+.*?from\s+['\"](.+?)['\"]", content)
        
        stats = []
        if classes:
            stats.append(f"{len(classes)} class(es)")
        if functions:
            stats.append(f"{len(set(functions))} function(s)")
        if imports:
            stats.append(f"{len(imports)} import(s)")
        
        if stats:
            lines.append(f"📊 Contains: {', '.join(stats)}")
        
        if exports:
            lines.append(f"\n📤 Exports: {', '.join(exports[:5])}")
        
        return "\n".join(lines)
    
    def _summarize_markdown(self, content: str, filename: str) -> str:
        """Summarize Markdown file."""
        lines = [f"📄 {filename}"]
        
        # Extract headers
        headers = re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)
        
        # Get first paragraph
        paragraphs = re.split(r"\n\s*\n", content)
        first_para = ""
        for p in paragraphs:
            p = p.strip()
            if p and not p.startswith("#") and not p.startswith("```"):
                first_para = p[:200]
                break
        
        if first_para:
            lines.append(f"\n{first_para}...")
        
        if headers:
            lines.append("\n📑 Structure:")
            for level, title in headers[:8]:
                indent = "  " * (len(level) - 1)
                lines.append(f"   {indent}• {title}")
            if len(headers) > 8:
                lines.append(f"   ... and {len(headers) - 8} more sections")
        
        return "\n".join(lines)
    
    def _summarize_text(self, content: str, filename: str) -> str:
        """Summarize generic text file."""
        lines = [f"📄 {filename}"]
        
        total_lines = content.count("\n") + 1
        total_words = len(content.split())
        total_chars = len(content)
        
        lines.append(f"📊 Size: {total_lines} lines, {total_words} words, {total_chars} characters")
        
        # First few non-empty lines as preview
        preview_lines = []
        for line in content.split("\n")[:10]:
            line = line.strip()
            if line:
                preview_lines.append(line[:80])
        
        if preview_lines:
            lines.append("\n📝 Preview:")
            for pl in preview_lines[:5]:
                lines.append(f"   {pl}")
        
        return "\n".join(lines)
