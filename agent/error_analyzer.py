"""
Error Analyzer - Analyzes execution errors and suggests fixes.
"""

import re
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum


class ErrorType(Enum):
    """Types of errors that can occur during execution."""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    NAME_ERROR = "name_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    INDEX_ERROR = "index_error"
    KEY_ERROR = "key_error"
    ATTRIBUTE_ERROR = "attribute_error"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT_ERROR = "timeout_error"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"


@dataclass
class ErrorAnalysis:
    """Result of error analysis."""
    error_type: ErrorType
    message: str
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    suggestion: str = ""
    context: str = ""
    fixable: bool = False
    fix_prompt: str = ""
    
    def to_prompt(self) -> str:
        """Convert to a prompt for the LLM to fix the error."""
        parts = [
            f"Error Type: {self.error_type.value}",
            f"Error Message: {self.message}",
        ]
        
        if self.line_number:
            parts.append(f"Line Number: {self.line_number}")
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        if self.context:
            parts.append(f"Context:\n{self.context}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        if self.fix_prompt:
            parts.append(f"\nFix Instructions: {self.fix_prompt}")
        
        return "\n".join(parts)


class ErrorAnalyzer:
    """
    Analyzes execution errors and provides actionable suggestions.
    
    This class parses error messages and tracebacks to:
    1. Identify the type of error
    2. Extract relevant context (line number, file, etc.)
    3. Suggest potential fixes
    4. Generate prompts for the LLM to fix errors
    """
    
    # Common Python error patterns
    PYTHON_PATTERNS = {
        ErrorType.SYNTAX_ERROR: [
            r"SyntaxError:\s*(.+)",
            r"IndentationError:\s*(.+)",
            r"TabError:\s*(.+)",
        ],
        ErrorType.IMPORT_ERROR: [
            r"ModuleNotFoundError:\s*No module named\s*['\"](.+)['\"]",
            r"ImportError:\s*(.+)",
        ],
        ErrorType.NAME_ERROR: [
            r"NameError:\s*name\s*['\"](.+)['\"]\s*is not defined",
        ],
        ErrorType.TYPE_ERROR: [
            r"TypeError:\s*(.+)",
        ],
        ErrorType.VALUE_ERROR: [
            r"ValueError:\s*(.+)",
        ],
        ErrorType.INDEX_ERROR: [
            r"IndexError:\s*(.+)",
        ],
        ErrorType.KEY_ERROR: [
            r"KeyError:\s*(.+)",
        ],
        ErrorType.ATTRIBUTE_ERROR: [
            r"AttributeError:\s*(.+)",
        ],
        ErrorType.FILE_NOT_FOUND: [
            r"FileNotFoundError:\s*(.+)",
            r"No such file or directory:\s*['\"](.+)['\"]",
        ],
        ErrorType.PERMISSION_ERROR: [
            r"PermissionError:\s*(.+)",
        ],
    }
    
    # Line number extraction patterns
    LINE_PATTERNS = [
        r'File "(.+)", line (\d+)',
        r"line (\d+)",
        r":(\d+):",
    ]
    
    def analyze(self, error_output: str, code: Optional[str] = None) -> ErrorAnalysis:
        """
        Analyze an error output and return analysis.
        
        Args:
            error_output: stderr or error message from execution
            code: Optional source code for context
            
        Returns:
            ErrorAnalysis with details and suggestions
        """
        # Detect error type
        error_type = ErrorType.UNKNOWN
        message = error_output
        
        for err_type, patterns in self.PYTHON_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, error_output, re.IGNORECASE)
                if match:
                    error_type = err_type
                    message = match.group(1) if match.groups() else error_output
                    break
            if error_type != ErrorType.UNKNOWN:
                break
        
        # Extract line number and file
        line_number = None
        file_path = None
        
        for pattern in self.LINE_PATTERNS:
            match = re.search(pattern, error_output)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    file_path = groups[0]
                    line_number = int(groups[1])
                elif len(groups) == 1:
                    line_number = int(groups[0])
                break
        
        # Extract context from code
        context = ""
        if code and line_number:
            lines = code.split("\n")
            start = max(0, line_number - 3)
            end = min(len(lines), line_number + 2)
            context_lines = []
            for i in range(start, end):
                prefix = ">>> " if i == line_number - 1 else "    "
                context_lines.append(f"{prefix}{i+1}: {lines[i]}")
            context = "\n".join(context_lines)
        
        # Generate suggestion
        suggestion = self._get_suggestion(error_type, message)
        
        # Generate fix prompt
        fix_prompt = self._get_fix_prompt(error_type, message, context)
        
        # Determine if fixable
        fixable = error_type not in (ErrorType.TIMEOUT_ERROR, ErrorType.PERMISSION_ERROR)
        
        return ErrorAnalysis(
            error_type=error_type,
            message=message,
            line_number=line_number,
            file_path=file_path,
            suggestion=suggestion,
            context=context,
            fixable=fixable,
            fix_prompt=fix_prompt,
        )
    
    def _get_suggestion(self, error_type: ErrorType, message: str) -> str:
        """Get a human-readable suggestion for the error."""
        suggestions = {
            ErrorType.SYNTAX_ERROR: (
                "Check for missing colons, parentheses, brackets, or quotation marks. "
                "Also check indentation is consistent (use spaces or tabs, not both)."
            ),
            ErrorType.IMPORT_ERROR: (
                "The module may need to be installed. Try: pip install <module_name>. "
                "Or check if the module name is spelled correctly."
            ),
            ErrorType.NAME_ERROR: (
                "The variable or function is used before it's defined. "
                "Check spelling and ensure it's defined in the correct scope."
            ),
            ErrorType.TYPE_ERROR: (
                "An operation was performed on incompatible types. "
                "Check argument types and consider type conversion."
            ),
            ErrorType.VALUE_ERROR: (
                "A function received an argument with the right type but wrong value. "
                "Validate input values before using them."
            ),
            ErrorType.INDEX_ERROR: (
                "Trying to access an index that doesn't exist. "
                "Check list/array length and loop boundaries."
            ),
            ErrorType.KEY_ERROR: (
                "Trying to access a dictionary key that doesn't exist. "
                "Use .get() method or check if key exists first."
            ),
            ErrorType.ATTRIBUTE_ERROR: (
                "Trying to access an attribute that doesn't exist on the object. "
                "Check the object type and available attributes/methods."
            ),
            ErrorType.FILE_NOT_FOUND: (
                "The specified file or path doesn't exist. "
                "Check the path and create the file/directory if needed."
            ),
            ErrorType.PERMISSION_ERROR: (
                "Don't have permission to access the file/resource. "
                "Check file permissions or run with appropriate privileges."
            ),
            ErrorType.TIMEOUT_ERROR: (
                "The operation took too long. "
                "Consider optimizing the code or increasing the timeout."
            ),
        }
        
        return suggestions.get(error_type, "Review the error message and check the code for issues.")
    
    def _get_fix_prompt(self, error_type: ErrorType, message: str, context: str) -> str:
        """Generate a prompt to help the LLM fix the error."""
        prompts = {
            ErrorType.SYNTAX_ERROR: (
                "Fix the syntax error in the code. Look for:\n"
                "- Missing or extra parentheses, brackets, or braces\n"
                "- Missing colons after if/for/while/def/class statements\n"
                "- Incorrect indentation\n"
                "- Unclosed strings"
            ),
            ErrorType.IMPORT_ERROR: (
                f"The import failed: {message}\n"
                "Either:\n"
                "1. Install the missing package if it's a third-party library\n"
                "2. Fix the import path if it's a local module\n"
                "3. Remove the import if it's not needed"
            ),
            ErrorType.NAME_ERROR: (
                f"Variable or function not defined: {message}\n"
                "Fix by:\n"
                "1. Define the variable/function before use\n"
                "2. Fix the spelling if it's a typo\n"
                "3. Import it if it's from another module"
            ),
            ErrorType.TYPE_ERROR: (
                f"Type error: {message}\n"
                "Fix by:\n"
                "1. Convert types appropriately (str(), int(), float(), etc.)\n"
                "2. Check function argument types\n"
                "3. Verify the object type before calling methods"
            ),
            ErrorType.INDEX_ERROR: (
                "Array index out of bounds. Fix by:\n"
                "1. Check the array length before accessing\n"
                "2. Use try-except for safe access\n"
                "3. Fix loop range to stay within bounds"
            ),
            ErrorType.KEY_ERROR: (
                f"Dictionary key not found: {message}\n"
                "Fix by:\n"
                "1. Use dict.get(key, default) for safe access\n"
                "2. Check if key exists with 'if key in dict'\n"
                "3. Ensure the key is added before access"
            ),
            ErrorType.ATTRIBUTE_ERROR: (
                f"Attribute error: {message}\n"
                "Fix by:\n"
                "1. Check the object type (print type(obj))\n"
                "2. Verify the attribute/method name spelling\n"
                "3. Ensure the object is properly initialized"
            ),
            ErrorType.FILE_NOT_FOUND: (
                f"File not found: {message}\n"
                "Fix by:\n"
                "1. Create the file/directory if it should exist\n"
                "2. Fix the path if it's incorrect\n"
                "3. Check if the file exists before accessing"
            ),
        }
        
        base_prompt = prompts.get(error_type, f"Fix the following error: {message}")
        
        if context:
            base_prompt += f"\n\nError context:\n{context}"
        
        return base_prompt


class AutoFixer:
    """
    Automatic code fixer that works with the LLM.
    
    This class coordinates between error analysis and the LLM to
    automatically fix code issues.
    """
    
    def __init__(self, max_attempts: int = 3):
        self.analyzer = ErrorAnalyzer()
        self.max_attempts = max_attempts
        self.attempt_count = 0
        self.fix_history: list[dict] = []
    
    def should_attempt_fix(self, error_output: str, code: str) -> tuple[bool, Optional[ErrorAnalysis]]:
        """
        Determine if we should attempt to fix the error.
        
        Returns:
            Tuple of (should_fix, analysis)
        """
        if self.attempt_count >= self.max_attempts:
            return False, None
        
        analysis = self.analyzer.analyze(error_output, code)
        
        if not analysis.fixable:
            return False, analysis
        
        # Check if we've tried this exact fix before
        fix_key = f"{analysis.error_type}:{analysis.message}"
        if fix_key in [h.get("key") for h in self.fix_history]:
            return False, analysis
        
        return True, analysis
    
    def record_attempt(self, analysis: ErrorAnalysis, fixed_code: str, success: bool):
        """Record a fix attempt."""
        self.attempt_count += 1
        self.fix_history.append({
            "key": f"{analysis.error_type}:{analysis.message}",
            "error_type": analysis.error_type,
            "success": success,
            "attempt": self.attempt_count,
        })
    
    def reset(self):
        """Reset the fixer state for a new task."""
        self.attempt_count = 0
        self.fix_history = []
    
    def get_fix_context(self, analysis: ErrorAnalysis) -> str:
        """Get context for the LLM to fix the error."""
        return (
            f"Please fix the following error:\n\n"
            f"{analysis.to_prompt()}\n\n"
            f"Provide the corrected code that resolves this error."
        )
