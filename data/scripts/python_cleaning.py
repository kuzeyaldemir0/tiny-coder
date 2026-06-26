from __future__ import annotations

import ast
import builtins
import io
import keyword
import re
import tokenize
from urllib.request import urlretrieve

from .constants import CJK_THRESHOLD, EN_CONF_MIN, MIN_MIXED_TEXT_CHARS, MODEL_PATH, MODEL_URL, SOURCE_PYTHON
from .io_utils import text_hash


CJK_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    return len(CJK_PATTERN.findall(text)) / len(text)


def parse_python3(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


BOILERPLATE_COMMENT_PATTERNS = [
    re.compile(r"^\s*#\s*-\*-\s*coding[:=]\s*[-\w.]+\s*-\*-\s*$", re.I),
    re.compile(r"^\s*#.*\bcoding[:=]\s*[-\w.]+.*$", re.I),
    re.compile(r"^\s*#!.*\bpython[\w.]*\b.*$", re.I),
    re.compile(r"^\s*#\s*(created on|created by|author|@author|date|@date|last modified|modified by)\b.*$", re.I),
    re.compile(
        r"^\s*#\s*(file|filename|name|course|class|section|assignment|term|instructor|professor|teacher|student|std\d*|program|desc|description|usage)\s*[:\-].*$",
        re.I,
    ),
    re.compile(r"^\s*#\s*[-=#*_]{4,}\s*$"),
]

BOILERPLATE_DOCSTRING_PATTERNS = [
    re.compile(r"^\s*(created on|created by|author|@author|date|@date|last modified|modified by)\b.*$", re.I),
    re.compile(
        r"^\s*(file|filename|name|course|class|section|assignment|term|instructor|professor|teacher|student|std\d*|program|desc|description|usage)\s*[:\-].*$",
        re.I,
    ),
    re.compile(r"^\s*[-=#*_]{4,}\s*$"),
    re.compile(r"^\s*(editor|editeur|éditeur)\s+de\s+spyder\b.*$", re.I),
    re.compile(r"^\s*this is a temporary script\.?\s*$", re.I),
    re.compile(r"^\s*ceci est un script temporaire\.?\s*$", re.I),
]

BOILERPLATE_CONTINUATION_PATTERN = re.compile(r"^\s*#\s{2,}\S.*$")


def is_boilerplate_comment_line(line: str) -> bool:
    return any(pattern.match(line) for pattern in BOILERPLATE_COMMENT_PATTERNS)


def is_boilerplate_docstring_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(pattern.match(stripped) for pattern in BOILERPLATE_DOCSTRING_PATTERNS)


def remove_leading_comment_boilerplate(lines: list[str]) -> tuple[list[str], int]:
    cleaned = list(lines)
    removed = 0
    last_removed_metadata = False

    for i, line in enumerate(cleaned[:40]):
        stripped = line.strip()
        if not stripped:
            last_removed_metadata = False
            continue
        if stripped.startswith("#") or stripped.startswith("#!"):
            is_boilerplate = is_boilerplate_comment_line(line)
            is_continuation = last_removed_metadata and BOILERPLATE_CONTINUATION_PATTERN.match(line)
            if is_boilerplate or is_continuation:
                cleaned[i] = ""
                removed += 1
                last_removed_metadata = True
            else:
                last_removed_metadata = False
            continue
        break

    return cleaned, removed


def remove_boilerplate_module_docstring(code: str) -> tuple[str, int]:
    tree = parse_python3(code)
    if tree is None or not tree.body:
        return code, 0

    first = tree.body[0]
    if not (
        isinstance(first, ast.Expr)
        and isinstance(getattr(first, "value", None), ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return code, 0

    doc_lines = first.value.value.splitlines()
    if not doc_lines or not all(is_boilerplate_docstring_line(line) for line in doc_lines):
        return code, 0

    lines = code.splitlines(keepends=True)
    start = first.lineno - 1
    end = getattr(first, "end_lineno", first.lineno)
    removed = max(0, end - start)
    del lines[start:end]
    return "".join(lines), removed


def strip_known_boilerplate(code: str) -> tuple[str, dict]:
    original = code
    lines = code.splitlines(keepends=True)
    lines, comment_lines_removed = remove_leading_comment_boilerplate(lines)
    code = "".join(lines).lstrip("\n")
    code, docstring_lines_removed = remove_boilerplate_module_docstring(code)
    code = code.lstrip("\n")

    if parse_python3(code) is None:
        return original, {
            "changed": False,
            "chars_removed": 0,
            "lines_removed": 0,
            "reparse_failed": True,
        }

    return code, {
        "changed": code != original,
        "chars_removed": len(original) - len(code),
        "lines_removed": comment_lines_removed + docstring_lines_removed,
        "reparse_failed": False,
    }


PYTHON_NAMES = set(keyword.kwlist) | set(dir(builtins)) | {
    "self",
    "cls",
    "args",
    "kwargs",
    "main",
    "none",
    "true",
    "false",
}


def extract_nl(code: str) -> str:
    nl: list[str] = []
    strings: list[str] = []
    prev = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                nl.append(tok.string.lstrip("#").strip())
            elif tok.type == tokenize.STRING and prev in (
                tokenize.INDENT,
                tokenize.NEWLINE,
                tokenize.NL,
                None,
            ):
                strings.append(tok.string)
            if tok.type != tokenize.NL:
                prev = tok.type
    except (tokenize.TokenError, IndentationError):
        pass
    return " ".join(nl + strings).strip()


def split_identifier(name: str) -> list[str]:
    name = name.strip("_")
    if not name or (name.startswith("__") and name.endswith("__")):
        return []

    words: list[str] = []
    for part in re.split(r"_+", name):
        words.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", part))

    return [
        w.lower()
        for w in words
        if len(w) >= 3 and not w.isdigit() and w.lower() not in PYTHON_NAMES
    ]


class IdentifierVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []
        self.imported: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imported.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imported.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.append(node.name)
        self._visit_arguments(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.append(node.name)
        self._visit_arguments(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.append(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.append(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Store):
            if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
                self.names.append(node.attr)
        self.generic_visit(node)

    def _visit_arguments(self, args: ast.arguments) -> None:
        all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        if args.vararg:
            all_args.append(args.vararg)
        if args.kwarg:
            all_args.append(args.kwarg)
        for arg in all_args:
            self.names.append(arg.arg)


def extract_identifier_text(code_or_tree: str | ast.AST) -> str:
    tree = code_or_tree if isinstance(code_or_tree, ast.AST) else ast.parse(code_or_tree)
    visitor = IdentifierVisitor()
    visitor.visit(tree)

    words: list[str] = []
    for name in visitor.names:
        if name in visitor.imported or name in PYTHON_NAMES:
            continue
        words.extend(split_identifier(name))
    return " ".join(words)


_LID_MODEL = None


def get_lid_model():
    global _LID_MODEL
    if _LID_MODEL is None:
        import fasttext

        if not MODEL_PATH.exists():
            urlretrieve(MODEL_URL, MODEL_PATH)
        _LID_MODEL = fasttext.load_model(str(MODEL_PATH))
    return _LID_MODEL


def detect_lang(text: str) -> tuple[str, float]:
    text = text.replace("\n", " ").strip()
    if not text:
        return "unknown", 0.0
    labels, probs = get_lid_model().predict(text, k=1)
    return labels[0].replace("__label__", ""), float(probs[0])


def clean_python_row(row: dict, require_fasttext: bool = True) -> tuple[dict | None, str | None, dict]:
    text = row.get("text") or ""
    meta = {"boilerplate_changed": False, "boilerplate_chars_removed": 0}
    if cjk_fraction(text) > CJK_THRESHOLD:
        return None, "cjk", meta

    tree = parse_python3(text)
    if tree is None:
        return None, "not_python3", meta

    cleaned, info = strip_known_boilerplate(text)
    meta["boilerplate_changed"] = bool(info["changed"])
    meta["boilerplate_chars_removed"] = int(info["chars_removed"])
    if info["reparse_failed"]:
        return None, "boilerplate_reparse_failed", meta

    text = cleaned
    tree = parse_python3(text)
    if tree is None:
        return None, "not_python3_after_boilerplate", meta

    nl_text = extract_nl(text)
    identifier_text = extract_identifier_text(tree)
    mixed_text = f"{nl_text} {identifier_text}".strip()
    if len(mixed_text) < MIN_MIXED_TEXT_CHARS:
        return None, "short_language_signal", meta

    lang = "unknown"
    lang_conf = 0.0
    if require_fasttext:
        lang, lang_conf = detect_lang(mixed_text)
        if lang != "en" or lang_conf < EN_CONF_MIN:
            return None, "not_english", meta

    row_id = row.get("blob_id") or row.get("id") or row.get("path") or text_hash(text)
    cleaned_row = {
        "id": str(row_id),
        "source": SOURCE_PYTHON,
        "text": text,
        "path": row.get("path", ""),
        "repo_name": row.get("repo_name", ""),
        "lang": lang,
        "lang_conf": lang_conf,
        "mixed_chars": len(mixed_text),
        "nl_chars": len(nl_text),
        "identifier_chars": len(identifier_text),
    }
    return cleaned_row, None, meta
