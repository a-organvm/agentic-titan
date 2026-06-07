"""
Local Trainer - Learn from user's local codebase patterns.

Provides:
- Pattern extraction from codebases
- Style adaptation for code generation
- Local fine-tuning support
- Coding convention learning

Reference: vendor/agents/igor/ local learning patterns
"""

from __future__ import annotations

import ast
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger("titan.learning.local_trainer")


class PatternType(StrEnum):
    NAMING = "naming"
    STRUCTURE = "structure"
    STYLE = "style"
    IDIOM = "idiom"
    IMPORT = "import"
    DOCSTRING = "docstring"
    ERROR_HANDLING = "error_handling"
    TYPING = "typing"


@dataclass
class CodingPattern:
    type: PatternType
    name: str
    description: str
    frequency: int = 1
    examples: list[str] = field(default_factory=list)
    confidence: float = 1.0
    language: str = "python"
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "frequency": self.frequency,
            "examples": self.examples[:5],
            "confidence": self.confidence,
            "language": self.language,
        }


@dataclass
class StyleProfile:
    variable_case: str = "snake_case"
    function_case: str = "snake_case"
    class_case: str = "PascalCase"
    constant_case: str = "UPPER_SNAKE_CASE"
    private_prefix: str = "_"
    indent_size: int = 4
    max_line_length: int = 88
    quote_style: str = "double"
    trailing_comma: bool = True
    import_order: list[str] = field(default_factory=lambda: ["stdlib", "third_party", "local"])
    import_style: str = "grouped"
    docstring_style: str = "google"
    inline_comments: bool = True
    type_annotations: bool = True
    exception_style: str = "specific"
    error_messages: str = "descriptive"
    patterns: list[CodingPattern] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "naming": {
                "variable_case": self.variable_case,
                "function_case": self.function_case,
                "class_case": self.class_case,
                "constant_case": self.constant_case,
                "private_prefix": self.private_prefix,
            },
            "formatting": {
                "indent_size": self.indent_size,
                "max_line_length": self.max_line_length,
                "quote_style": self.quote_style,
                "trailing_comma": self.trailing_comma,
            },
            "patterns": [p.to_dict() for p in self.patterns[:20]],
        }

    def to_prompt_context(self) -> str:
        lines = ["# Coding Style Guide", ""]
        lines.extend(
            [
                "## Naming Conventions",
                f"- Variables: {self.variable_case}",
                f"- Functions: {self.function_case}",
                f"- Classes: {self.class_case}",
                f"- Constants: {self.constant_case}",
                f"- Private members: prefix with '{self.private_prefix}'",
                "",
            ]
        )
        if self.patterns:
            lines.extend(["## Common Patterns", ""])
            for pattern in self.patterns[:10]:
                lines.append(f"- {pattern.name}: {pattern.description}")
        return "\n".join(lines)


@dataclass
class TrainingConfig:
    source_dirs: list[str] = field(default_factory=list)
    include_patterns: list[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["*test*", "*__pycache__*", "*.pyc"]
    )
    min_file_size: int = 100
    max_file_size: int = 1_000_000
    min_examples: int = 3
    learn_naming: bool = True
    learn_structure: bool = True
    learn_idioms: bool = True
    learn_docstrings: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"min_examples": self.min_examples}


@dataclass
class TrainingResult:
    style_profile: StyleProfile
    files_analyzed: int = 0
    patterns_found: int = 0
    training_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_analyzed": self.files_analyzed,
            "patterns_found": self.patterns_found,
            "style_profile": self.style_profile.to_dict(),
        }


class PatternExtractor:
    def __init__(self) -> None:
        self._naming_stats: Counter[str] = Counter()
        self._patterns: list[CodingPattern] = []

    def analyze_file(self, filepath: Path) -> list[CodingPattern]:
        try:
            content = filepath.read_text(encoding="utf-8")
            tree = ast.parse(content)
            naming = self._extract_naming_patterns(tree, filepath)
            idioms = self._extract_idiom_patterns(tree, filepath)
            file_patterns = naming + idioms
            self._patterns.extend(file_patterns)
            return file_patterns
        except Exception as e:
            logger.debug(f"Failed to parse {filepath}: {e}")
            return []

    def _extract_naming_patterns(self, tree: ast.AST, filepath: Path) -> list[CodingPattern]:
        patterns = []
        function_cases: Counter[str] = Counter()
        class_cases: Counter[str] = Counter()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                case = self._detect_case(node.name)
                self._naming_stats[f"function:{case}"] += 1
                function_cases[case] += 1
            elif isinstance(node, ast.ClassDef):
                case = self._detect_case(node.name)
                self._naming_stats[f"class:{case}"] += 1
                class_cases[case] += 1

        for case, count in function_cases.items():
            patterns.append(
                CodingPattern(
                    type=PatternType.NAMING,
                    name=f"function_{case}",
                    description=f"Uses {case} for function names",
                    frequency=count,
                    source_files=[str(filepath)],
                )
            )
        for case, count in class_cases.items():
            patterns.append(
                CodingPattern(
                    type=PatternType.NAMING,
                    name=f"class_{case}",
                    description=f"Uses {case} for class names",
                    frequency=count,
                    source_files=[str(filepath)],
                )
            )
        return patterns

    def _extract_idiom_patterns(self, tree: ast.AST, filepath: Path) -> list[CodingPattern]:
        patterns = []
        nodes = list(ast.walk(tree))
        list_comp_count = sum(1 for node in nodes if isinstance(node, ast.ListComp))
        if list_comp_count > 0:
            patterns.append(
                CodingPattern(
                    type=PatternType.IDIOM,
                    name="list_comprehension",
                    description="Uses list comprehensions",
                    frequency=list_comp_count,
                    source_files=[str(filepath)],
                )
            )
        annotated_functions = [
            node
            for node in nodes
            if isinstance(node, ast.FunctionDef)
            and (
                node.returns is not None
                or any(arg.annotation is not None for arg in node.args.args)
                or any(arg.annotation is not None for arg in node.args.kwonlyargs)
            )
        ]
        if annotated_functions:
            patterns.append(
                CodingPattern(
                    type=PatternType.TYPING,
                    name="type_annotations",
                    description="Uses function type annotations",
                    frequency=len(annotated_functions),
                    source_files=[str(filepath)],
                )
            )
        return patterns

    def _detect_case(self, name: str) -> str:
        if name.isupper():
            return "UPPER_SNAKE_CASE"
        if "_" in name:
            return "snake_case"
        if name[0].isupper():
            return "PascalCase"
        return "snake_case"

    def get_style_profile(self, min_frequency: int = 3) -> StyleProfile:
        profile = StyleProfile()
        for prefix in ["function", "class"]:
            cases = {
                k.split(":")[1]: v
                for k, v in self._naming_stats.items()
                if k.startswith(f"{prefix}:")
            }
            if cases:
                setattr(profile, f"{prefix}_case", max(cases, key=lambda case: cases[case]))

        profile.patterns = [p for p in self._patterns if p.frequency >= min_frequency]
        return profile


class StyleAdapter:
    def __init__(self, style_profile: StyleProfile) -> None:
        self.style_profile = style_profile

    def adapt_code(self, code: str) -> str:
        if self.style_profile.quote_style == "single":
            return code.replace('"', "'")
        return code

    def suggest_improvements(self, code: str) -> list[str]:
        suggestions = []
        if "def " in code and self.style_profile.function_case == "camelCase" and "_" in code:
            suggestions.append("Function uses snake_case, but project style is camelCase")
        return suggestions

    def get_prompt_context(self) -> str:
        return self.style_profile.to_prompt_context()


class LocalTrainer:
    def __init__(
        self,
        config: TrainingConfig | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._style_profile: StyleProfile | None = None
        self._extractor = PatternExtractor()

    def train(self, source_path: str | Path) -> TrainingResult:
        import time
        start_time = time.time()
        source_path = Path(source_path)
        files_analyzed = 0
        for filepath in source_path.rglob("*.py"):
            if "test" in filepath.name:
                continue
            try:
                size = filepath.stat().st_size
                if size < self.config.min_file_size or size > self.config.max_file_size:
                    continue
            except OSError:
                continue
            self._extractor.analyze_file(filepath)
            files_analyzed += 1
        self._style_profile = self._extractor.get_style_profile(
            min_frequency=self.config.min_examples
        )
        return TrainingResult(
            style_profile=self._style_profile,
            files_analyzed=files_analyzed,
            patterns_found=len(self._style_profile.patterns),
            training_time_ms=(time.time() - start_time) * 1000,
        )

    def get_adapter(self) -> StyleAdapter:
        return StyleAdapter(self._style_profile or StyleProfile())


def extract_patterns(
    source_path: str | Path,
    config: TrainingConfig | None = None,
) -> list[CodingPattern]:
    trainer = LocalTrainer(config=config)
    result = trainer.train(source_path)
    return result.style_profile.patterns


def train_on_codebase(
    source_path: str | Path,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    trainer = LocalTrainer(config=config)
    return trainer.train(source_path)


def get_style_context(source_path: str | Path) -> str:
    trainer = LocalTrainer()
    trainer.train(source_path)
    return trainer.get_adapter().get_prompt_context()


_trainer: LocalTrainer | None = None


def get_trainer() -> LocalTrainer:
    global _trainer
    if _trainer is None:
        _trainer = LocalTrainer()
    return _trainer
