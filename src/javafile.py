"""Static metrics for a single new .java file (used by score.score_new_file).

Each metric is one of:
  computed -- from the file itself (size, comments, complexity, methods, fields, nesting, coupling, cohesion)
  assumed  -- new-file defaults for process/ownership (1 commit, 1 author, 100% new code)
  imputed  -- needs whole-codebase/git context; filled with the training release median
"""
import javalang


def _filter_any(node, types):
    """node.filter() for several types (javalang only accepts one)."""
    for t in types:
        yield from node.filter(t)


# ============================================================================= LINE-LEVEL
def _line_kinds(text):
    """Label each line "code", "comment" or "blank" (handles multi-line /* */)."""
    kinds = []
    in_block = False   # inside an open /* */
    for raw in text.splitlines():
        line = raw.strip()
        if in_block:
            kinds.append("comment")
            if "*/" in line:
                in_block = False
                rest = line.split("*/", 1)[-1].strip()
                if rest and not rest.startswith("//"):
                    kinds[-1] = "code"   # code after closing */
            continue
        if not line:
            kinds.append("blank"); continue
        if line.startswith("//"):
            kinds.append("comment"); continue
        if line.startswith("/*"):
            kinds.append("comment")
            if "*/" not in line:
                in_block = True
            continue
        kinds.append("code")
    return kinds


# ============================================================================= AST HELPERS
# nodes that add +1 to cyclomatic complexity (switch cases and &&/|| handled separately)
DECISION_TYPES = (javalang.tree.IfStatement, javalang.tree.ForStatement, javalang.tree.WhileStatement,
                   javalang.tree.DoStatement, javalang.tree.CatchClause, javalang.tree.TernaryExpression)


def _cyclomatic(method_node, collapse_switch=False):
    """McCabe complexity of a method. collapse_switch=True counts a switch once ("Modified")."""
    c = 1
    for _, _ in _filter_any(method_node, DECISION_TYPES):
        c += 1
    if collapse_switch:
        c += len(list(method_node.filter(javalang.tree.SwitchStatement)))       # +1 per switch
    else:
        c += len(list(method_node.filter(javalang.tree.SwitchStatementCase)))   # +1 per case
    for _, op in method_node.filter(javalang.tree.BinaryOperation):
        if op.operator in ("&&", "||"):
            c += 1
    return c


def _max_nesting(node, depth=0):
    """Deepest nesting of if/for/while/do/try/switch under `node`."""
    best = depth
    block_types = (javalang.tree.IfStatement, javalang.tree.ForStatement, javalang.tree.WhileStatement,
                   javalang.tree.DoStatement, javalang.tree.TryStatement, javalang.tree.SwitchStatement)
    children = getattr(node, "children", [])
    for c in _flatten(children):
        if isinstance(c, javalang.ast.Node):
            nd = depth + 1 if isinstance(c, block_types) else depth
            best = max(best, _max_nesting(c, nd))
    return best


def _flatten(x):
    """Yield AST nodes from nested lists/tuples/sets."""
    if isinstance(x, javalang.ast.Node):
        yield x
    elif isinstance(x, (list, tuple, set)):
        for e in x:
            yield from _flatten(e)


def _method_body_nodes(cls_node):
    """Methods/constructors declared directly in this class (not nested classes)."""
    out = []
    for decl in cls_node.body or []:
        if isinstance(decl, (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)):
            out.append(decl)
    return out


def _visibility_counts(methods):
    """(public, private, protected, package-default) method counts."""
    pub = sum(1 for m in methods if "public" in m.modifiers)
    priv = sum(1 for m in methods if "private" in m.modifiers)
    prot = sum(1 for m in methods if "protected" in m.modifiers)
    default = len(methods) - pub - priv - prot
    return pub, priv, prot, max(default, 0)


def _fields(cls_node):
    """(name, is_static) for each field in this class (handles `int a, b;`)."""
    fields = []
    for decl in cls_node.body or []:
        if isinstance(decl, javalang.tree.FieldDeclaration):
            for d in decl.declarators:
                fields.append((d.name, "static" in decl.modifiers))
    return fields


def _referenced_field_names(method_node, field_names):
    """Class fields used by this method (`this.x` included)."""
    used = set()
    for _, ref in method_node.filter(javalang.tree.MemberReference):
        if ref.member in field_names:
            used.add(ref.member)
    return used


def _lack_of_cohesion(methods, field_names):
    """% of method pairs sharing no field (0 = cohesive, 100 = none shared)."""
    if len(methods) < 2 or not field_names:
        return 0.0
    used = [_referenced_field_names(m, field_names) for m in methods]
    pairs = 0; sharing = 0
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            pairs += 1
            if used[i] & used[j]:
                sharing += 1
    if pairs == 0:
        return 0.0
    return round(100.0 * (pairs - sharing) / pairs, 2)


def _referenced_type_names(tree):
    """Distinct non-JDK type names referenced in the file (proxy for CountClassCoupled)."""
    names = set()
    for _, imp in tree.filter(javalang.tree.Import):
        names.add(imp.path.split(".")[-1])   # "com.foo.Bar" -> "Bar"
    for _, rt in tree.filter(javalang.tree.ReferenceType):
        if rt.name:
            names.add(rt.name)
    for _, cr in tree.filter(javalang.tree.ClassCreator):
        if cr.type and cr.type.name:
            names.add(cr.type.name)          # `new Foo()`
    return {n for n in names if n and n not in JDK_COMMON}


# common JDK types, excluded from coupling
JDK_COMMON = {"String", "Integer", "Long", "Double", "Float", "Boolean", "Object", "Exception",
              "RuntimeException", "Throwable", "List", "ArrayList", "Map", "HashMap", "Set", "HashSet",
              "Void", "Character", "Byte", "Short", "StringBuilder", "StringBuffer", "Iterator", "Iterable",
              "Comparable", "Collection", "Override", "Optional", "Stream", "Collectors"}


# ============================================================================= ENTRY POINTS
def analyze_java_source(text):
    """Return (computed, assumed) metric dicts for one Java source. Raises on invalid Java."""
    kinds = _line_kinds(text)
    n_code = kinds.count("code"); n_comment = kinds.count("comment"); n_blank = kinds.count("blank")
    n_line = len(kinds)

    tree = javalang.parse.parse(text)
    # aggregate over all classes/interfaces in the file
    classes = list(_filter_any(tree, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)))
    top_classes = [n for _, n in classes]
    all_methods, all_fields, extends_count, base_count = [], [], 0, 0
    for _, cls in classes:
        methods = _method_body_nodes(cls)
        all_methods += methods
        all_fields += _fields(cls)
        if getattr(cls, "extends", None):
            extends_count += 1; base_count += 1
        if getattr(cls, "implements", None):
            base_count += len(cls.implements)

    n_methods = max(len(all_methods), 1)   # avoid divide by zero
    complexities = [_cyclomatic(m) for m in all_methods] or [1]
    complexities_mod = [_cyclomatic(m, collapse_switch=True) for m in all_methods] or [1]
    pub, priv, prot, default = _visibility_counts(all_methods)
    field_names = {n for n, _ in all_fields}
    static_fields = sum(1 for _, s in all_fields if s)
    instance_fields = len(all_fields) - static_fields
    nestings = [_max_nesting(m) for m in all_methods] or [0]
    stmt_count = len(list(tree.filter(javalang.tree.Statement)))
    decl_stmt_count = len(list(tree.filter(javalang.tree.LocalVariableDeclaration))) + len(all_fields)

    computed = {
        "CountLine": n_line, "CountLineCode": n_code, "CountLineComment": n_comment, "CountLineBlank": n_blank,
        "RatioCommentToCode": round(n_comment / max(n_code, 1), 3),
        "AvgLine": round(n_line / n_methods, 1), "AvgLineCode": round(n_code / n_methods, 1),
        "AvgLineComment": round(n_comment / n_methods, 1), "AvgLineBlank": round(n_blank / n_methods, 1),
        "CountDeclClass": len(top_classes), "CountDeclMethod": len(all_methods),
        "CountDeclFunction": len(all_methods), "CountDeclInstanceMethod": len(all_methods),
        "CountDeclMethodPublic": pub, "CountDeclMethodPrivate": priv, "CountDeclMethodProtected": prot,
        "CountDeclMethodDefault": default, "CountDeclClassMethod": static_fields,
        "CountDeclClassVariable": static_fields, "CountDeclInstanceVariable": instance_fields,
        "SumCyclomatic": sum(complexities), "MaxCyclomatic": max(complexities),
        "AvgCyclomatic": round(sum(complexities) / n_methods, 2),
        # Strict = plain here (not distinguished by this analyzer)
        "SumCyclomaticStrict": sum(complexities), "MaxCyclomaticStrict": max(complexities),
        "AvgCyclomaticStrict": round(sum(complexities) / n_methods, 2),
        "SumCyclomaticModified": sum(complexities_mod), "MaxCyclomaticModified": max(complexities_mod),
        "AvgCyclomaticModified": round(sum(complexities_mod) / n_methods, 2),
        "MaxNesting_Max": max(nestings), "MaxNesting_Mean": round(sum(nestings) / n_methods, 2),
        "MaxNesting_Min": min(nestings),
        "CountStmt": stmt_count, "CountStmtDecl": decl_stmt_count,
        "CountStmtExe": max(stmt_count - decl_stmt_count, 0), "CountSemicolon": stmt_count,
        "CountClassCoupled": len(_referenced_type_names(tree)),
        "CountClassBase": base_count, "CountClassDerived": 0,   # new file: no subclasses yet
        "MaxInheritanceTree": 1 if extends_count else 0,       # only direct parent visible
        "PercentLackOfCohesion": _lack_of_cohesion(all_methods, field_names),
    }
    assumed = {   # new-file defaults
        "COMM": 1, "ADEV": 1, "DDEV": 1, "Added_lines": n_code, "Del_lines": 0,
        "OWN_LINE": 100.0, "OWN_COMMIT": 100.0, "MINOR_COMMIT": 0, "MINOR_LINE": 0,
        "MAJOR_COMMIT": 100.0, "MAJOR_LINE": 100.0,
    }
    return computed, assumed


def build_release_row(file_path, features, impute_medians):
    """Full feature row for a .java file: computed + assumed + imputed (median).
    Returns (row, computed_keys, assumed_keys, imputed_keys)."""
    text = open(file_path, encoding="utf-8", errors="replace").read()
    computed, assumed = analyze_java_source(text)
    row = {}
    for f in features:
        if f in computed:
            row[f] = computed[f]
        elif f in assumed:
            row[f] = assumed[f]
        else:
            row[f] = impute_medians.get(f, 0.0)
    imputed_keys = [f for f in features if f not in computed and f not in assumed]
    return row, sorted(set(features) & computed.keys()), sorted(set(features) & assumed.keys()), imputed_keys
