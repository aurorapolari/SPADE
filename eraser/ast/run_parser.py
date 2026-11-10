"""Extract variable occurrences without modifying source coordinates.

The original release's DFG visitors are retained. Parser grammars are installed
as Python packages rather than bundled generated C sources and shared objects.
"""
from functools import lru_cache
import keyword

from tree_sitter import Language, Parser

from .parser_folder.DFG_java import DFG_java
from .parser_folder.DFG_python import DFG_python

JAVA_SPECIAL_IDS = set("main args Math System Random Byte Short Integer Long Float Double Character Boolean Data ParseException SimpleDateFormat Calendar Object String StringBuffer StringBuilder DateFormat Collection List Map Set Queue ArrayList HashSet HashMap".split())


@lru_cache(maxsize=2)
def get_parser(lang):
    if lang == "python":
        import tree_sitter_python as grammar
    elif lang == "java":
        import tree_sitter_java as grammar
    else:
        raise ValueError("Supported languages: python, java")
    return Parser(Language(grammar.language()))


def identifier_spans(code, lang, func_name=""):
    """Return (name, character_start, character_end) for DFG occurrences.

    Tree-sitter uses UTF-8 byte positions; tokenizers use character offsets.
    Comments and blank lines stay in place, and Java method fragments retain
    their original offsets even when parsed without an enclosing class.
    """
    source = code.encode("utf-8")
    root = get_parser(lang).parse(source).root_node
    leaves = []

    def visit(node):
        if not node.children or node.type == "string":
            leaves.append(node)
        else:
            for child in node.children:
                visit(child)

    visit(root)
    index_to_code = {
        (node.start_point, node.end_point): (i, source[node.start_byte:node.end_byte].decode("utf-8"))
        for i, node in enumerate(leaves)
    }
    visitor = DFG_python if lang == "python" else DFG_java
    dfg, _ = visitor(root, index_to_code, {})
    result = {}
    for name, index, *_ in dfg:
        if name == func_name or not name.isidentifier() or (lang == "python" and keyword.iskeyword(name)):
            continue
        if lang == "java" and name in JAVA_SPECIAL_IDS:
            continue
        node = leaves[index]
        start = len(source[:node.start_byte].decode("utf-8"))
        end = len(source[:node.end_byte].decode("utf-8"))
        result[(start, end)] = (name, start, end)
    return [result[key] for key in sorted(result)]
