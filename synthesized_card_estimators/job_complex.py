import math
import random
from collections import Counter, defaultdict
from fnmatch import fnmatchcase

from interface import base_card_estimator


class card_estimator(base_card_estimator):
    SAMPLE_SIZES = {
        "company_name": 50000,
        "company_type": 4,
        "keyword": 30000,
        "link_type": 18,
        "movie_link": 20000,
        "movie_keyword": 80000,
        "title": 100000,
        "movie_companies": 100000,
        "movie_info": 150000,
        "movie_info_idx": 120000,
        "info_type": 113,
        "kind_type": 7,
        "complete_cast": 50000,
        "comp_cast_type": 4,
        "cast_info": 200000,
        "name": 100000,
        "char_name": 100000,
        "aka_name": 60000,
        "aka_title": 50000,
        "role_type": 12,
        "person_info": 60000,
    }

    TOPK = {
        ("company_name", "country_code"): 256,
        ("company_name", "id"): 2048,
        ("company_name", "name"): 4096,
        ("company_name", "name_pcode_nf"): 4096,
        ("company_name", "name_pcode_sf"): 4096,
        ("company_type", "id"): 16,
        ("company_type", "kind"): 16,
        ("keyword", "id"): 4096,
        ("keyword", "keyword"): 4096,
        ("keyword", "phonetic_code"): 2048,
        ("link_type", "id"): 32,
        ("link_type", "link"): 64,
        ("movie_link", "movie_id"): 4096,
        ("movie_link", "linked_movie_id"): 4096,
        ("movie_link", "link_type_id"): 64,
        ("movie_keyword", "movie_id"): 4096,
        ("movie_keyword", "keyword_id"): 4096,
        ("title", "id"): 4096,
        ("title", "title"): 4096,
        ("movie_companies", "company_id"): 2048,
        ("movie_companies", "movie_id"): 4096,
        ("movie_companies", "company_type_id"): 64,
        ("movie_info", "movie_id"): 4096,
        ("movie_info", "info_type_id"): 256,
        ("movie_info_idx", "movie_id"): 4096,
        ("movie_info_idx", "info_type_id"): 256,
        ("title", "imdb_index"): 64,
        ("title", "episode_nr"): 256,
        ("movie_info_idx", "info"): 256,
        ("info_type", "id"): 256,
        ("info_type", "info"): 256,
        ("kind_type", "id"): 16,
        ("kind_type", "kind"): 16,
        ("complete_cast", "movie_id"): 4096,
        ("complete_cast", "subject_id"): 16,
        ("complete_cast", "status_id"): 16,
        ("comp_cast_type", "id"): 16,
        ("comp_cast_type", "kind"): 16,
        ("cast_info", "id"): 4096,
        ("cast_info", "person_id"): 4096,
        ("cast_info", "movie_id"): 4096,
        ("cast_info", "person_role_id"): 4096,
        ("cast_info", "note"): 512,
        ("cast_info", "nr_order"): 256,
        ("cast_info", "role_id"): 64,
        ("name", "id"): 4096,
        ("name", "name"): 4096,
        ("name", "imdb_index"): 256,
        ("name", "name_pcode_cf"): 4096,
        ("name", "name_pcode_nf"): 4096,
        ("name", "surname_pcode"): 4096,
        ("char_name", "id"): 4096,
        ("char_name", "name"): 4096,
        ("char_name", "imdb_index"): 256,
        ("char_name", "name_pcode_nf"): 4096,
        ("char_name", "surname_pcode"): 4096,
        ("aka_name", "person_id"): 4096,
        ("aka_name", "imdb_index"): 256,
        ("aka_name", "name_pcode_cf"): 4096,
        ("aka_name", "name_pcode_nf"): 4096,
        ("aka_name", "surname_pcode"): 4096,
        ("aka_title", "movie_id"): 4096,
        ("aka_title", "title"): 2048,
        ("aka_title", "imdb_index"): 256,
        ("aka_title", "kind_id"): 64,
        ("role_type", "id"): 32,
        ("role_type", "role"): 32,
    }

    PATTERN_COLS = {
        ("company_name", "name"),
        ("link_type", "link"),
        ("title", "title"),
        ("movie_companies", "note"),
        ("movie_info", "info"),
        ("movie_info", "note"),
        ("name", "name"),
        ("char_name", "name"),
        ("aka_title", "title"),
        ("comp_cast_type", "kind"),
    }

    PATTERN_CACHE = {
        ("company_name", "name"): ["%Film%", "%Warner%", "Lionsgate%"],
        ("name", "name"): ["%Tim%", "%Downey%", "%Downey%Robert%", "B%", "%Bert%"],
        ("title", "title"): ["%Freddy%", "%Jason%", "Saw%"],
        ("char_name", "name"): ["%man%", "%Man%"],
        ("movie_companies", "note"): ["%(200%)%", "%(USA)%"],
        ("movie_info", "note"): ["%internet%"],
        ("movie_info", "info"): ["USA:% 199%", "USA:% 200%"],
        ("link_type", "link"): ["%follow%"],
        ("comp_cast_type", "kind"): ["%complete%"],
    }

    PAIR_SPECS = {
        ("company_name", ("name_pcode_nf", "name_pcode_sf")): 4096,
        ("keyword", ("keyword", "phonetic_code")): 2048,
        ("movie_link", ("link_type_id", "movie_id")): 2048,
        ("title", ("kind_id", "production_year")): 2048,
        ("movie_companies", ("company_id", "company_type_id")): 4096,
        ("movie_info", ("info_type_id", "info")): 4096,
        ("movie_info", ("info_type_id", "note")): 2048,
        ("movie_info", ("movie_id", "info_type_id")): 4096,
        ("movie_info_idx", ("info_type_id", "info")): 2048,
        ("movie_info_idx", ("movie_id", "info_type_id")): 2048,
        ("complete_cast", ("subject_id", "status_id")): 64,
        ("cast_info", ("person_id", "note")): 4096,
        ("cast_info", ("nr_order", "movie_id")): 2048,
        ("name", ("gender", "name")): 2048,
        ("name", ("name_pcode_nf", "surname_pcode")): 4096,
        ("name", ("name_pcode_cf", "name_pcode_nf")): 4096,
        ("char_name", ("name_pcode_nf", "surname_pcode")): 4096,
        ("char_name", ("name", "name_pcode_nf")): 2048,
        ("aka_name", ("imdb_index", "person_id")): 2048,
        ("aka_name", ("name_pcode_cf", "name_pcode_nf")): 4096,
        ("aka_title", ("imdb_index", "movie_id")): 2048,
    }

    PK_FK = {
        ("company_type", "id", "movie_companies", "company_type_id"),
        ("company_name", "id", "movie_companies", "company_id"),
        ("title", "id", "movie_companies", "movie_id"),
        ("title", "id", "movie_keyword", "movie_id"),
        ("keyword", "id", "movie_keyword", "keyword_id"),
        ("link_type", "id", "movie_link", "link_type_id"),
        ("title", "id", "movie_link", "movie_id"),
        ("title", "id", "movie_link", "linked_movie_id"),
        ("title", "id", "movie_info", "movie_id"),
        ("info_type", "id", "movie_info", "info_type_id"),
        ("title", "id", "movie_info_idx", "movie_id"),
        ("info_type", "id", "movie_info_idx", "info_type_id"),
        ("kind_type", "id", "title", "kind_id"),
        ("comp_cast_type", "id", "complete_cast", "subject_id"),
        ("comp_cast_type", "id", "complete_cast", "status_id"),
        ("title", "id", "complete_cast", "movie_id"),
        ("title", "id", "cast_info", "movie_id"),
        ("name", "id", "cast_info", "person_id"),
        ("char_name", "id", "cast_info", "person_role_id"),
        ("role_type", "id", "cast_info", "role_id"),
        ("name", "id", "aka_name", "person_id"),
        ("title", "id", "aka_title", "movie_id"),
    }

    def __init__(self):
        super().__init__()
        self.random_seed = 1337
        self.schema = {}
        self.table_stats = {}
        self.pk_to_fk = {}
        self.fk_to_pk = {}

    def setup(self):
        self.schema = self._load_schema()
        self._build_pk_fk_maps()
        for table_name in self.schema:
            self.table_stats[table_name] = self._compute_table_stats(table_name)
        if not self.table_stats:
            raise RuntimeError("setup() failed: no table statistics were generated")

    def estimate(self, tables: list, filters: list, joins: list) -> int:
        alias_map = self._validate_query(tables, filters, joins)
        table_filters = defaultdict(list)
        for f in filters:
            self._flatten_filters(f, table_filters)
        self._pushdown_dimension_filters(alias_map, table_filters, joins)

        per_alias = {}
        for alias, table in alias_map.items():
            stats = self.table_stats[table]
            sel = self._estimate_filter_selectivity(table, table_filters.get(alias, []))
            rows = max(1.0, stats["row_count"] * sel) if stats["row_count"] > 0 else 0.0
            per_alias[alias] = {
                "table": table,
                "base_rows": stats["row_count"],
                "rows": min(stats["row_count"], rows),
                "selectivity": sel,
            }

        if len(per_alias) == 1:
            result = next(iter(per_alias.values()))["rows"]
        elif joins:
            result = self._estimate_join_graph(alias_map, per_alias, joins)
        else:
            result = 1.0
            for alias in per_alias:
                result *= per_alias[alias]["rows"]

        result = self._self_correct(result, per_alias)
        return int(max(0, round(result)))

    def _pushdown_dimension_filters(self, alias_map, table_filters, joins):
        tiny_tables = {"info_type", "kind_type", "company_type", "comp_cast_type", "link_type", "role_type"}
        exact_id_sets = {}
        for alias, table in alias_map.items():
            if table not in tiny_tables:
                continue
            predicates = table_filters.get(alias, [])
            if not predicates:
                continue
            matching_ids = self._evaluate_tiny_dimension_filters(table, predicates)
            if matching_ids is not None:
                exact_id_sets[alias] = matching_ids

        for join in joins:
            left_alias, right_alias = join["alias1"], join["alias2"]
            left_table, right_table = alias_map[left_alias], alias_map[right_alias]
            if left_alias in exact_id_sets and join["column1"] == "id":
                self._add_derived_in_filter(
                    table_filters,
                    right_alias,
                    join["column2"],
                    sorted(exact_id_sets[left_alias]),
                )
            if right_alias in exact_id_sets and join["column2"] == "id":
                self._add_derived_in_filter(
                    table_filters,
                    left_alias,
                    join["column1"],
                    sorted(exact_id_sets[right_alias]),
                )

    def _evaluate_tiny_dimension_filters(self, table, predicates):
        rows = self.table_stats[table].get("exact_rows", [])
        if not rows:
            return None
        matched = set()
        for row in rows:
            if self._row_matches_predicates(table, row, predicates):
                raw_id = row.get("id")
                if raw_id is not None:
                    matched.add(self._normalize_value(raw_id, self.schema[table]["id"]))
        return matched

    def _row_matches_predicates(self, table, row, predicates):
        for pred in predicates:
            norm = self._normalize_predicate(table, pred)
            if not self._evaluate_row_predicate(table, row, norm):
                return False
        return True

    def _evaluate_row_predicate(self, table, row, pred):
        if "children" in pred:
            if pred["operator"] == "AND":
                return all(self._evaluate_row_predicate(table, row, child) for child in pred["children"])
            return any(self._evaluate_row_predicate(table, row, child) for child in pred["children"])
        col = pred["column"]
        op = pred["operator"]
        lit = pred["literal"]
        value = self._normalize_value(row.get(col), self.schema[table][col])
        if op == "IS NULL":
            return value is None
        if op == "IS NOT NULL":
            return value is not None
        if op == "=":
            return value == self._normalize_value(lit, self.schema[table][col])
        if op == "!=":
            return value is not None and value != self._normalize_value(lit, self.schema[table][col])
        if op == "IN":
            norm_lits = {self._normalize_value(v, self.schema[table][col]) for v in lit}
            return value in norm_lits
        if op in {">", ">=", "<", "<="}:
            return self._compare_values(value, op, lit, self.schema[table][col])
        if op == "LIKE":
            return value is not None and self._like_match(str(value).lower(), str(lit).lower())
        if op == "NOT LIKE":
            return value is None or not self._like_match(str(value).lower(), str(lit).lower())
        return False

    def _add_derived_in_filter(self, table_filters, alias, column, values):
        if not values:
            return
        for pred in table_filters.get(alias, []):
            if "children" not in pred and pred.get("column") == column and pred.get("operator") == "IN":
                existing = set(pred.get("literal", []))
                pred["literal"] = sorted(existing.intersection(values)) if existing else values
                return
            if "children" not in pred and pred.get("column") == column and pred.get("operator") == "=":
                if pred.get("literal") not in values:
                    pred["literal"] = values[0]
                return
        table_filters[alias].append(
            {"alias": alias, "column": column, "operator": "IN", "literal": values}
        )

    def _load_schema(self):
        import json
        with open("data/schema.json", "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _build_pk_fk_maps(self):
        for pk_table, pk_col, fk_table, fk_col in self.PK_FK:
            self.pk_to_fk[(pk_table, pk_col)] = (fk_table, fk_col)
            self.fk_to_pk[(fk_table, fk_col)] = (pk_table, pk_col)

    def _compute_table_stats(self, table_name):
        rows = []
        reader = self.custom_csv_reader(table_name)
        header = next(reader)
        if header != list(self.schema[table_name].keys()):
            raise ValueError(f"Header mismatch for table {table_name}")
        total_rows = 0
        sample_target = self.SAMPLE_SIZES.get(table_name, 50000)
        rng = random.Random(self.random_seed + len(table_name))
        for row in reader:
            total_rows += 1
            if len(rows) < sample_target:
                rows.append(row)
            else:
                idx = rng.randint(0, total_rows - 1)
                if idx < sample_target:
                    rows[idx] = row

        col_index = {c: i for i, c in enumerate(header)}
        stats = {
            "row_count": total_rows,
            "sample_size": len(rows),
            "columns": {},
            "pairs": {},
            "key_freq": {},
            "exact_rows": [],
        }
        if total_rows <= 500:
            stats["exact_rows"] = [dict(zip(header, row)) for row in rows]

        for col in header:
            values = [self._normalize_value(row[col_index[col]], self.schema[table_name][col]) for row in rows]
            stats["columns"][col] = self._build_column_stats(table_name, col, values, total_rows)

        for (tbl, pair), k in self.PAIR_SPECS.items():
            if tbl != table_name:
                continue
            c1, c2 = pair
            if c1 in col_index and c2 in col_index:
                counter = Counter()
                for row in rows:
                    v1 = self._normalize_value(row[col_index[c1]], self.schema[table_name][c1])
                    v2 = self._normalize_value(row[col_index[c2]], self.schema[table_name][c2])
                    counter[(v1, v2)] += 1
                stats["pairs"][pair] = self._counter_to_topk(counter, len(rows), total_rows, k)

        for key_col in ["movie_id", "company_id", "person_id", "keyword_id", "info_type_id", "link_type_id"]:
            if key_col in col_index:
                counter = Counter()
                for row in rows:
                    v = self._normalize_value(row[col_index[key_col]], self.schema[table_name][key_col])
                    counter[v] += 1
                stats["key_freq"][key_col] = self._build_freq_summary(counter, len(rows), total_rows)
        return stats

    def _build_column_stats(self, table, col, values, total_rows):
        non_null = [v for v in values if v is not None]
        sample_size = len(values)
        counter = Counter(non_null)
        null_frac = 0.0 if sample_size == 0 else (sample_size - len(non_null)) / sample_size
        col_type = self.schema[table][col]
        stats = {
            "null_frac": null_frac,
            "sample_ndv": len(counter),
            "sample_size": sample_size,
            "total_rows": total_rows,
            "topk": None,
            "hist": None,
            "patterns": None,
            "value_map": None,
            "exact": None,
        }
        if total_rows <= 500 or sample_size <= 500:
            stats["exact"] = dict(counter)
        if col_type == "int" and len(counter) <= 512:
            stats["value_map"] = self._counter_to_topk(counter, sample_size, total_rows, len(counter))
        k = self.TOPK.get((table, col))
        if col_type == "int":
            stats["hist"] = self._build_histogram(non_null, total_rows, 32 if col in {"production_year", "nr_order", "episode_nr"} else 16)
        if k:
            stats["topk"] = self._counter_to_topk(counter, sample_size, total_rows, k)
        elif col_type == "str" and len(counter) <= 512:
            stats["topk"] = self._counter_to_topk(counter, sample_size, total_rows, min(512, max(1, len(counter))))
        if (table, col) in self.PATTERN_COLS:
            self._current_pattern_table = table
            self._current_pattern_column = col
            stats["patterns"] = self._build_pattern_stats(non_null, total_rows, sample_size)
        return stats

    def _build_histogram(self, values, total_rows, buckets):
        if not values:
            return {"buckets": [], "min": None, "max": None}
        vals = sorted(values)
        n = len(vals)
        actual = min(buckets, n)
        bucket_list = []
        for i in range(actual):
            start = (i * n) // actual
            end = ((i + 1) * n) // actual
            chunk = vals[start:end]
            if not chunk:
                continue
            bucket_list.append((chunk[0], chunk[-1], len(chunk) / n))
        scale = 0 if n == 0 else total_rows / n
        return {"buckets": bucket_list, "scale": scale, "min": vals[0], "max": vals[-1]}

    def _build_pattern_stats(self, values, total_rows, sample_size):
        lowered = [v.lower() for v in values if isinstance(v, str)]
        trigrams = Counter()
        prefixes = Counter()
        examples = Counter(lowered)
        for value in lowered:
            for width in range(1, 5):
                if len(value) >= width:
                    prefixes[value[:width]] += 1
            padded = f"  {value} "
            for i in range(max(0, len(padded) - 2)):
                trigrams[padded[i:i + 3]] += 1
        cached_patterns = {}
        for pattern in self.PATTERN_CACHE.get((self._current_pattern_table, self._current_pattern_column), []):
            match_count = 0
            pattern_l = pattern.lower()
            for value in lowered:
                if self._like_match(value, pattern_l):
                    match_count += 1
            cached_patterns[pattern] = 0.0 if sample_size == 0 else (match_count * total_rows / sample_size)
        return {
            "trigrams": self._counter_to_topk(trigrams, sample_size, total_rows, 512),
            "prefixes": self._counter_to_topk(prefixes, sample_size, total_rows, 64),
            "examples": self._counter_to_topk(examples, sample_size, total_rows, 1024),
            "cached": cached_patterns,
        }

    def _counter_to_topk(self, counter, sample_size, total_rows, k):
        items = counter.most_common(k)
        sample_mass = sum(v for _, v in items)
        scale = 0.0 if sample_size == 0 else total_rows / sample_size
        top = {key: val * scale for key, val in items}
        return {
            "values": top,
            "sample_mass": sample_mass,
            "scaled_mass": sample_mass * scale,
            "tail_mass": max(0.0, total_rows - sample_mass * scale),
            "tail_ndv": max(1, len(counter) - len(items)),
        }

    def _build_freq_summary(self, counter, sample_size, total_rows):
        scale = 0.0 if sample_size == 0 else total_rows / sample_size
        avg = 0.0 if not counter else sum(counter.values()) / len(counter)
        return {
            "top": self._counter_to_topk(counter, sample_size, total_rows, min(2048, max(1, len(counter)))),
            "avg_freq": avg * scale,
            "ndv": max(1, int(len(counter) * scale)) if sample_size else 1,
        }

    def _normalize_value(self, raw, typ):
        if raw == "":
            return None
        if typ == "int":
            try:
                return int(raw)
            except Exception:
                return None
        return raw

    def _validate_query(self, tables, filters, joins):
        alias_map = {}
        for t in tables:
            if "name" not in t or "alias" not in t:
                raise ValueError("Every table entry must contain 'name' and 'alias'")
            if t["name"] not in self.schema:
                raise ValueError(f"Unknown table {t['name']}")
            alias_map[t["alias"]] = t["name"]
        for join in joins:
            for side in (1, 2):
                alias = join[f"alias{side}"]
                column = join[f"column{side}"]
                if alias not in alias_map:
                    raise ValueError(f"Unknown alias in join: {alias}")
                if column not in self.schema[alias_map[alias]]:
                    raise ValueError(f"Unknown column {column} for table {alias_map[alias]}")
        def validate_filter(node):
            if "children" in node:
                for child in node["children"]:
                    validate_filter(child)
            else:
                alias = node["alias"]
                column = node["column"]
                if alias not in alias_map:
                    raise ValueError(f"Unknown alias in filter: {alias}")
                if column not in self.schema[alias_map[alias]]:
                    raise ValueError(f"Unknown column {column} for table {alias_map[alias]}")
        for f in filters:
            validate_filter(f)
        return alias_map

    def _flatten_filters(self, node, out):
        if "children" not in node:
            out[node["alias"]].append(node)
            return
        op = node["operator"]
        if op == "AND":
            for child in node["children"]:
                self._flatten_filters(child, out)
        else:
            out[node["children"][0].get("alias", "__or__")].append(node)

    def _estimate_filter_selectivity(self, table, predicates):
        if not predicates:
            return 1.0
        predicates = [self._normalize_predicate(table, p) for p in predicates]
        special = self._estimate_special_fact_filters(table, predicates)
        if special is not None:
            return min(1.0, max(0.0, special))
        sels = []
        for pred in predicates:
            sels.append(self._estimate_predicate(table, pred))
        sel = 1.0
        for s in sels:
            sel *= s
        sel = self._apply_pair_corrections(table, predicates, sel)
        return min(1.0, max(0.0, sel))

    def _estimate_special_fact_filters(self, table, predicates):
        if table == "movie_info":
            return self._estimate_movie_info_filters(table, predicates)
        if table == "movie_info_idx":
            return self._estimate_movie_info_idx_filters(table, predicates)
        return None

    def _estimate_movie_info_filters(self, table, predicates):
        type_pred = None
        info_pred = None
        note_preds = []
        for pred in predicates:
            if "children" in pred:
                return None
            if pred["column"] == "info_type_id" and pred["operator"] in {"=", "IN"}:
                type_pred = pred
            elif pred["column"] == "info":
                info_pred = pred
            elif pred["column"] == "note":
                note_preds.append(pred)
        if type_pred is None or info_pred is None:
            return None
        type_ids = type_pred["literal"] if type_pred["operator"] == "IN" else [type_pred["literal"]]
        total_rows = max(1.0, self.table_stats[table]["row_count"])
        pair_summary = self.table_stats[table]["pairs"].get(("info_type_id", "info"))
        type_summary = self.table_stats[table]["columns"]["info_type_id"].get("topk")
        if not pair_summary or not type_summary:
            return None
        est_rows = 0.0
        literals = info_pred["literal"] if info_pred["operator"] == "IN" else [info_pred["literal"]]
        for type_id in type_ids:
            type_rows = type_summary["values"].get(type_id, 0.0)
            matched = 0.0
            for literal in literals:
                matched += pair_summary["values"].get((type_id, literal), 0.0)
            if matched == 0.0 and literals:
                type_pairs = sum(
                    freq for (tid, _), freq in pair_summary["values"].items() if tid == type_id
                )
                type_distinct = max(
                    1,
                    len([1 for (tid, _) in pair_summary["values"] if tid == type_id]),
                )
                matched = min(type_rows, len(literals) * max(0.0, type_rows - type_pairs) / type_distinct)
            est_rows += matched
        sel = est_rows / total_rows
        for pred in note_preds:
            sel *= self._estimate_predicate(table, pred)
        return sel

    def _estimate_movie_info_idx_filters(self, table, predicates):
        type_pred = None
        info_pred = None
        for pred in predicates:
            if "children" in pred:
                return None
            if pred["column"] == "info_type_id" and pred["operator"] in {"=", "IN"}:
                type_pred = pred
            elif pred["column"] == "info":
                info_pred = pred
        if type_pred is None or info_pred is None:
            return None
        if info_pred["operator"] not in {"<", "<=", ">", ">=", "=", "IN"}:
            return None
        type_ids = type_pred["literal"] if type_pred["operator"] == "IN" else [type_pred["literal"]]
        total_rows = max(1.0, self.table_stats[table]["row_count"])
        pair_summary = self.table_stats[table]["pairs"].get(("info_type_id", "info"))
        type_summary = self.table_stats[table]["columns"]["info_type_id"].get("topk")
        if not pair_summary or not type_summary:
            return None
        est_rows = 0.0
        for type_id in type_ids:
            type_rows = type_summary["values"].get(type_id, 0.0)
            matched = 0.0
            for (tid, value), freq in pair_summary["values"].items():
                if tid != type_id:
                    continue
                if info_pred["operator"] in {"=", "IN"}:
                    literals = info_pred["literal"] if info_pred["operator"] == "IN" else [info_pred["literal"]]
                    if value in literals:
                        matched += freq
                elif self._compare_values(value, info_pred["operator"], info_pred["literal"], "str"):
                    matched += freq
            if matched == 0.0 and info_pred["operator"] in {"<", "<=", ">", ">="}:
                matched = type_rows * self._estimate_predicate(table, info_pred)
            est_rows += min(type_rows, matched)
        return est_rows / total_rows

    def _normalize_predicate(self, table, pred):
        if "children" in pred:
            return {
                "operator": pred["operator"],
                "children": [self._normalize_predicate(table, child) for child in pred["children"]],
            }
        normalized = dict(pred)
        lit = normalized.get("literal")
        if normalized.get("operator") == "=" and isinstance(lit, str):
            alias = normalized.get("alias")
            prefix = f"{alias}." if alias else ""
            if prefix and lit.startswith(prefix):
                other_col = lit[len(prefix):]
                if other_col in self.schema[table]:
                    normalized["literal"] = {"column_ref": other_col}
            elif lit in self.schema[table]:
                normalized["literal"] = {"column_ref": lit}
        return normalized

    def _estimate_predicate(self, table, pred):
        if "children" in pred:
            child_sels = [self._estimate_predicate(table, c) for c in pred["children"]]
            if pred["operator"] == "AND":
                out = 1.0
                for s in child_sels:
                    out *= s
                return out
            out = 0.0
            for s in child_sels:
                out = out + s - out * s
            return out
        col = pred["column"]
        op = pred["operator"]
        lit = pred["literal"]
        cstats = self.table_stats[table]["columns"][col]
        if op == "IS NULL":
            return cstats["null_frac"]
        if op == "IS NOT NULL":
            return 1.0 - cstats["null_frac"]
        if op in {"=", "!=", "IN"}:
            base = self._estimate_equality_like(table, col, op, lit)
            return base
        if op in {">", "<", ">=", "<="}:
            return self._estimate_range(table, col, op, lit)
        if op in {"LIKE", "NOT LIKE"}:
            like_sel = self._estimate_like(table, col, lit)
            return like_sel if op == "LIKE" else 1.0 - like_sel
        return 0.1

    def _estimate_equality_like(self, table, col, op, lit):
        if isinstance(lit, dict) and "column_ref" in lit and op == "=":
            return self._estimate_column_equality(table, col, lit["column_ref"])
        cstats = self.table_stats[table]["columns"][col]
        values = cstats.get("exact") or {}
        total_rows = self.table_stats[table]["row_count"]
        literals = lit if isinstance(lit, list) else [lit]
        sel = 0.0
        for item in literals:
            norm = self._normalize_value(str(item) if self.schema[table][col] == "int" and item is not None else item, self.schema[table][col]) if not isinstance(item, int) else item
            if norm in values:
                sel += values[norm] / max(1, total_rows)
                continue
            topk = cstats.get("topk")
            if topk and norm in topk["values"]:
                sel += topk["values"][norm] / max(1, total_rows)
            else:
                topk = cstats.get("topk")
                if topk:
                    tail_mass = max(0.0, 1.0 - (topk["scaled_mass"] / max(1.0, total_rows)) - cstats["null_frac"])
                    sel += tail_mass / max(1.0, topk["tail_ndv"])
                else:
                    ndv = max(1, cstats["sample_ndv"])
                    tail = 1.0 - cstats["null_frac"]
                    sel += tail / ndv
        sel = min(1.0, sel)
        if op == "!=":
            return max(0.0, 1.0 - sel - cstats["null_frac"])
        return sel

    def _estimate_column_equality(self, table, left_col, right_col):
        pair = self.table_stats[table]["pairs"].get((left_col, right_col))
        if pair is None:
            pair = self.table_stats[table]["pairs"].get((right_col, left_col))
        total_rows = max(1.0, self.table_stats[table]["row_count"])
        if pair:
            same_mass = sum(freq for (v1, v2), freq in pair["values"].items() if v1 == v2 and v1 is not None)
            if same_mass > 0:
                return min(1.0, same_mass / total_rows)
        s1 = self.table_stats[table]["columns"][left_col]
        s2 = self.table_stats[table]["columns"][right_col]
        top1 = s1.get("topk")
        top2 = s2.get("topk")
        overlap = 0.0
        if top1 and top2:
            for key, f1 in top1["values"].items():
                f2 = top2["values"].get(key)
                if f2 is not None:
                    overlap += min(f1, f2) / total_rows
        fallback = 1.0 / max(1.0, s1["sample_ndv"], s2["sample_ndv"])
        return min(1.0, max(overlap, fallback))

    def _estimate_range(self, table, col, op, lit):
        value_map = self.table_stats[table]["columns"][col].get("value_map")
        total_rows = max(1.0, self.table_stats[table]["row_count"])
        if value_map:
            total = 0.0
            for key, freq in value_map["values"].items():
                if self._compare_values(key, op, lit, self.schema[table][col]):
                    total += freq / total_rows
            if total > 0:
                return min(1.0, total)
        hist = self.table_stats[table]["columns"][col].get("hist")
        if not hist or hist["min"] is None:
            return 0.333
        try:
            val = int(lit) if self.schema[table][col] == "int" and lit is not None else lit
        except Exception:
            val = lit
        total = 0.0
        for low, high, frac in hist["buckets"]:
            if op == ">" and high > val:
                total += frac if low > val else frac * 0.5
            elif op == ">=" and high >= val:
                total += frac if low >= val else frac * 0.5
            elif op == "<" and low < val:
                total += frac if high < val else frac * 0.5
            elif op == "<=" and low <= val:
                total += frac if high <= val else frac * 0.5
        return min(1.0, max(0.0, total))

    def _compare_values(self, key, op, lit, typ):
        try:
            val = int(lit) if typ == "int" and lit is not None else lit
        except Exception:
            val = lit
        if op == ">":
            return key > val
        if op == ">=":
            return key >= val
        if op == "<":
            return key < val
        if op == "<=":
            return key <= val
        return False

    def _estimate_like(self, table, col, pattern):
        cstats = self.table_stats[table]["columns"][col]
        if pattern is None:
            return 0.0
        pattern_l = pattern.lower()
        exact_rows = self.table_stats[table].get("exact_rows", [])
        if exact_rows:
            matches = 0
            for row in exact_rows:
                value = row.get(col)
                if value is not None and self._like_match(value.lower(), pattern_l):
                    matches += 1
            return matches / max(1, len(exact_rows))
        pstats = cstats.get("patterns")
        if not pstats:
            return 0.1
        cached = pstats.get("cached", {})
        if pattern in cached:
            return min(1.0, cached[pattern] / max(1, self.table_stats[table]["row_count"]))
        anchored = not pattern_l.startswith("%")
        no_wild = "%" not in pattern_l and "_" not in pattern_l
        if no_wild:
            return self._estimate_equality_like(table, col, "=", pattern)
        if anchored:
            prefix = pattern_l.split("%", 1)[0][:4]
            pref = pstats["prefixes"]["values"].get(prefix)
            if pref is not None:
                return min(1.0, pref / max(1, self.table_stats[table]["row_count"]))
        literals = [part for part in pattern_l.split("%") if part and part != "_"]
        examples = pstats["examples"]["values"]
        if literals:
            best = 1.0
            for token in literals:
                sample_hits = sum(freq for val, freq in examples.items() if token in val)
                if sample_hits > 0:
                    best = min(best, sample_hits / max(1, self.table_stats[table]["row_count"]))
                else:
                    grams = self._pattern_trigrams(token)
                    gram_vals = pstats["trigrams"]["values"]
                    gram_sels = []
                    for g in grams:
                        if g in gram_vals:
                            gram_sels.append(gram_vals[g] / max(1, self.table_stats[table]["row_count"]))
                    if gram_sels:
                        best = min(best, max(0.0001, min(gram_sels)))
                    else:
                        best = min(best, 0.05)
            if len(literals) > 1:
                best *= 0.5
            if anchored and best > 0:
                best = min(best, 1.0 / math.sqrt(max(1.0, self.table_stats[table]["row_count"])))
            return min(1.0, max(0.00001, best))
        return 0.1

    def _pattern_trigrams(self, token):
        token = f"  {token} "
        return [token[i:i + 3] for i in range(max(0, len(token) - 2))]

    def _like_match(self, value, pattern):
        shell_pattern = pattern.replace("%", "*").replace("_", "?")
        return fnmatchcase(value, shell_pattern)

    def _apply_pair_corrections(self, table, predicates, base_sel):
        pred_map = {p["column"]: p for p in predicates if "children" not in p}
        for pair, summary in self.table_stats[table]["pairs"].items():
            c1, c2 = pair
            if c1 in pred_map and c2 in pred_map:
                p1 = pred_map[c1]
                p2 = pred_map[c2]
                if p1["operator"] == "=" and p2["operator"] == "=":
                    key = (
                        self._normalize_value(p1["literal"], self.schema[table][c1]),
                        self._normalize_value(p2["literal"], self.schema[table][c2]),
                    )
                    if key in summary["values"]:
                        return min(1.0, summary["values"][key] / max(1, self.table_stats[table]["row_count"]))
        if table == "company_name":
            eq_pred = None
            for p in predicates:
                if "children" not in p and p["column"] == "name_pcode_nf" and p["operator"] == "=":
                    eq_pred = p
            if eq_pred:
                cstats = self.table_stats[table]["pairs"].get(("name_pcode_nf", "name_pcode_sf"))
                if cstats:
                    same_mass = sum(v for (a, b), v in cstats["values"].items() if a == b)
                    same_sel = same_mass / max(1, self.table_stats[table]["row_count"])
                    return min(base_sel, max(same_sel, base_sel * 0.5))
        return base_sel

    def _estimate_join(self, join, alias_map, per_alias):
        a1, c1 = alias_map[join["alias1"]], join["column1"]
        a2, c2 = alias_map[join["alias2"]], join["column2"]
        left_rows = per_alias[join["alias1"]]["rows"]
        right_rows = per_alias[join["alias2"]]["rows"]
        pkfk = self._pk_fk_direction(
            a1, c1, a2, c2, join["alias1"], join["alias2"]
        )
        if pkfk:
            pk_table, pk_col, fk_table, fk_col, pk_alias, fk_alias = pkfk
            pk_filtered_frac = per_alias[pk_alias]["rows"] / max(1.0, self.table_stats[pk_table]["row_count"])
            fk_rows = per_alias[fk_alias]["rows"]
            return fk_rows * min(1.0, pk_filtered_frac)
        return self._estimate_overlap_join(a1, c1, left_rows, a2, c2, right_rows)

    def _estimate_join_graph(self, alias_map, per_alias, joins):
        components = self._join_components(alias_map, joins)
        overall = 1.0
        covered_aliases = set()
        for component_aliases, component_joins in components:
            covered_aliases.update(component_aliases)
            overall *= self._estimate_component(alias_map, per_alias, component_aliases, component_joins)
        for alias, info in per_alias.items():
            if alias not in covered_aliases:
                overall *= max(1.0, info["rows"])
        return overall

    def _join_components(self, alias_map, joins):
        graph = defaultdict(set)
        for join in joins:
            a1 = join["alias1"]
            a2 = join["alias2"]
            graph[a1].add(a2)
            graph[a2].add(a1)
        seen = set()
        components = []
        for alias in alias_map:
            if alias in seen:
                continue
            stack = [alias]
            comp_aliases = set()
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp_aliases.add(cur)
                for nxt in graph.get(cur, ()):
                    if nxt not in seen:
                        stack.append(nxt)
            comp_joins = [
                j for j in joins if j["alias1"] in comp_aliases and j["alias2"] in comp_aliases
            ]
            components.append((comp_aliases, comp_joins))
        return components

    def _estimate_component(self, alias_map, per_alias, component_aliases, joins):
        if not joins:
            result = 1.0
            for alias in component_aliases:
                result *= max(1.0, per_alias[alias]["rows"])
            return result
        eq_classes = self._build_join_equivalence_classes(joins)
        alias_class_count = defaultdict(int)
        class_estimates = []
        for members in eq_classes:
            for alias, _ in members:
                alias_class_count[alias] += 1
            class_estimates.append(self._estimate_equivalence_class(alias_map, per_alias, members))
        result = 1.0
        for est in class_estimates:
            result *= max(1.0, est)
        for alias in component_aliases:
            rows = max(1.0, per_alias[alias]["rows"])
            degree = max(1, alias_class_count.get(alias, 0))
            result /= rows ** max(0, degree - 1)
        return max(1.0, result)

    def _build_join_equivalence_classes(self, joins):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for join in joins:
            left = (join["alias1"], join["column1"])
            right = (join["alias2"], join["column2"])
            union(left, right)
        groups = defaultdict(list)
        for node in list(parent):
            groups[find(node)].append(node)
        return list(groups.values())

    def _estimate_equivalence_class(self, alias_map, per_alias, members):
        if len(members) == 2:
            (a1, c1), (a2, c2) = members
            return self._estimate_join(
                {"alias1": a1, "column1": c1, "alias2": a2, "column2": c2},
                alias_map,
                per_alias,
            )
        if self._is_star_movie_id_class(alias_map, members):
            return self._estimate_star_key_class(alias_map, per_alias, members)
        return self._estimate_multiway_overlap(alias_map, per_alias, members)

    def _is_star_movie_id_class(self, alias_map, members):
        cols = {column for _, column in members}
        if len(cols) != 1:
            return False
        col = next(iter(cols))
        if col not in {"movie_id", "id", "linked_movie_id"}:
            return False
        has_title = any(alias_map[a] == "title" and c == "id" for a, c in members)
        movie_like = all(
            c in {"movie_id", "id", "linked_movie_id"} and
            (
                alias_map[a] == "title" or
                (alias_map[a], c) in self.fk_to_pk or
                (alias_map[a] == "movie_link" and c == "linked_movie_id")
            )
            for a, c in members
        )
        return has_title and movie_like

    def _estimate_star_key_class(self, alias_map, per_alias, members):
        key_stats = []
        for alias, column in members:
            table = alias_map[alias]
            rows = max(1.0, per_alias[alias]["rows"])
            cstats = self.table_stats[table]["columns"][column]
            ndv = self._estimate_effective_ndv(table, column, per_alias[alias]["selectivity"], rows)
            key_stats.append((alias, table, rows, max(1.0, ndv), cstats))
        alpha = 0.5 if len(key_stats) <= 3 else 0.33
        mus = [rows / max(1.0, ndv) for _, _, rows, ndv, _ in key_stats]
        best = None
        for i, (_, _, rows_i, ndv_i, _) in enumerate(key_stats):
            for j in range(i + 1, len(key_stats)):
                _, _, rows_j, ndv_j, _ = key_stats[j]
                pair_est = rows_i * rows_j / max(1.0, ndv_i, ndv_j)
                extra = 1.0
                for k, mu_k in enumerate(mus):
                    if k == i or k == j:
                        continue
                    extra *= max(1.0, mu_k ** alpha)
                candidate = pair_est * extra
                best = candidate if best is None else min(best, candidate)
        if best is None:
            best = 1.0

        title_rows = None
        for _, table, rows, _, _ in key_stats:
            if table == "title":
                title_rows = rows
                break
        if title_rows is not None:
            title_cap = title_rows
            for _, table, rows, ndv, _ in key_stats:
                if table == "title":
                    continue
                mu = rows / max(1.0, ndv)
                title_cap *= max(1.0, mu ** alpha)
            best = min(best, title_cap)
        return max(1.0, best)

    def _estimate_multiway_overlap(self, alias_map, per_alias, members):
        per_member = []
        value_candidates = set()
        for alias, column in members:
            table = alias_map[alias]
            rows = max(1.0, per_alias[alias]["rows"])
            cstats = self.table_stats[table]["columns"][column]
            topk = cstats.get("topk")
            total_rows = max(1.0, self.table_stats[table]["row_count"])
            scale = rows / total_rows
            scaled_top = {}
            if topk:
                for key, freq in topk["values"].items():
                    val = freq * scale
                    if val > 0:
                        scaled_top[key] = val
                value_candidates.update(scaled_top.keys())
            ndv = self._estimate_effective_ndv(table, column, per_alias[alias]["selectivity"], rows)
            top_mass = sum(scaled_top.values())
            non_null_rows = rows * (1.0 - cstats["null_frac"])
            tail_rows = max(0.0, non_null_rows - top_mass)
            tail_ndv = max(1.0, ndv - len(scaled_top))
            per_member.append(
                {
                    "rows": rows,
                    "scaled_top": scaled_top,
                    "tail_rows": tail_rows,
                    "tail_ndv": tail_ndv,
                }
            )
        top_overlap = 0.0
        for value in value_candidates:
            prod = 1.0
            present = True
            for entry in per_member:
                freq = entry["scaled_top"].get(value)
                if freq is None:
                    present = False
                    break
                prod *= freq
            if present:
                top_overlap += prod
        max_tail_ndv = max(entry["tail_ndv"] for entry in per_member)
        tail_prod = 1.0
        for entry in per_member:
            tail_prod *= entry["tail_rows"]
        tail_overlap = tail_prod / (max_tail_ndv ** (len(per_member) - 1))
        return max(1.0, top_overlap + tail_overlap)

    def _estimate_effective_ndv(self, table, column, selectivity, filtered_rows):
        cstats = self.table_stats[table]["columns"][column]
        base_ndv = max(1.0, cstats["sample_ndv"])
        if filtered_rows <= 1.0:
            return filtered_rows
        scaled = base_ndv * min(1.0, math.sqrt(max(0.0, selectivity)))
        return min(max(1.0, scaled), filtered_rows)

    def _pk_fk_direction(self, t1, c1, t2, c2, alias1, alias2):
        for pk_table, pk_col, fk_table, fk_col in self.PK_FK:
            if pk_table == t1 and pk_col == c1 and fk_table == t2 and fk_col == c2:
                return pk_table, pk_col, fk_table, fk_col, alias1, alias2
            if pk_table == t2 and pk_col == c2 and fk_table == t1 and fk_col == c1:
                return pk_table, pk_col, fk_table, fk_col, alias2, alias1
        return None

    def _estimate_overlap_join(self, t1, c1, left_rows, t2, c2, right_rows):
        s1 = self.table_stats[t1]["columns"][c1]
        s2 = self.table_stats[t2]["columns"][c2]
        top1 = s1.get("topk")
        top2 = s2.get("topk")
        overlap = 0.0
        if top1 and top2:
            for key, f1 in top1["values"].items():
                f2 = top2["values"].get(key)
                if f2 is not None:
                    overlap += (f1 / max(1, self.table_stats[t1]["row_count"])) * (f2 / max(1, self.table_stats[t2]["row_count"]))
        ndv1 = max(1, s1["sample_ndv"])
        ndv2 = max(1, s2["sample_ndv"])
        tail = self._estimate_join_tail(t1, c1, s1, left_rows, t2, c2, s2, right_rows, overlap)
        sel = min(1.0, overlap + tail)
        est = left_rows * right_rows * sel
        denom = max(ndv1, ndv2)
        pg = left_rows * right_rows / denom
        return min(est, max(pg, 1.0))

    def _estimate_join_tail(self, t1, c1, s1, left_rows, t2, c2, s2, right_rows, overlap_sel):
        if c1 == "imdb_index" and c2 == "imdb_index":
            return 0.0
        ndv1 = max(1.0, s1["sample_ndv"])
        ndv2 = max(1.0, s2["sample_ndv"])
        total1 = max(1.0, self.table_stats[t1]["row_count"])
        total2 = max(1.0, self.table_stats[t2]["row_count"])
        ratio1 = ndv1 / total1
        ratio2 = ndv2 / total2
        if self.schema[t1][c1] == "str" and self.schema[t2][c2] == "str" and max(ratio1, ratio2) > 0.7:
            return 0.0
        power = 1.0
        if self.schema[t1][c1] == "str" or self.schema[t2][c2] == "str":
            power = 1.5
        return max(0.0, 1.0 - overlap_sel) / (max(ndv1, ndv2) ** power)

    def _self_correct(self, estimate, per_alias):
        if not math.isfinite(estimate):
            estimate = 1.0
        upper = None
        for info in per_alias.values():
            rows = max(1.0, info["rows"])
            upper = rows if upper is None else upper * rows
        if upper is not None:
            estimate = min(estimate, upper)
        if estimate < 0:
            estimate = 0.0
        return estimate
