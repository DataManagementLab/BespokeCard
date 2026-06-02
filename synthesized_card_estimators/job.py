import math
import random
import re
from collections import Counter, defaultdict

from interface import base_card_estimator


class card_estimator(base_card_estimator):
    SAMPLE_SIZES = {
        "title": 250000,
        "cast_info": 400000,
        "char_name": 200000,
        "company_name": 120000,
        "company_type": 4,
        "movie_companies": 300000,
        "role_type": 12,
        "keyword": 80000,
        "link_type": 18,
        "movie_link": 30000,
        "movie_keyword": 300000,
        "info_type": 113,
        "movie_info": 350000,
        "movie_info_idx": 250000,
        "kind_type": 7,
        "aka_title": 120000,
        "aka_name": 150000,
        "name": 250000,
        "person_info": 180000,
        "complete_cast": 80000,
        "comp_cast_type": 4,
    }

    TEXT_CONFIG = {
        ("title", "title"): {"prefix": 2, "exact_k": 1024, "trigram_k": 4096},
        ("char_name", "name"): {"prefix": 2, "exact_k": 1024, "trigram_k": 4096},
        ("company_name", "name"): {"prefix": 3, "exact_k": 1024, "trigram_k": 2048},
        ("keyword", "keyword"): {"prefix": 0, "exact_k": 4096, "trigram_k": 2048},
        ("aka_name", "name"): {"prefix": 1, "exact_k": 512, "trigram_k": 2048},
        ("name", "name"): {"prefix": 1, "exact_k": 1024, "trigram_k": 4096},
    }

    TOKEN_CONFIG = {
        ("cast_info", "note"): [
            "(voice)",
            "(producer)",
            "(writer)",
            "(uncredited)",
            "producer",
            "voice",
        ],
        ("movie_companies", "note"): [
            "(usa)",
            "(worldwide)",
            "(japan)",
            "(blu-ray)",
            "(vhs)",
            "(tv)",
            "(co-production)",
            "(presents)",
            "(theatrical)",
            "mgm",
        ],
        ("movie_info", "note"): ["internet"],
    }

    CONDITIONAL_INFO_TABLES = {"movie_info", "movie_info_idx", "person_info"}

    CONDITIONAL_INFO_HINTS = {
        "movie_info": {
            "genres": {"horror", "action", "sci-fi", "thriller", "crime", "war"},
            "countries": {
                "bulgaria",
                "sweden",
                "norway",
                "germany",
                "denmark",
                "swedish",
                "denish",
                "norwegian",
                "german",
                "usa",
                "japan",
            },
        }
    }

    PK_FK = {
        ("aka_name", "person_id"): ("name", "id"),
        ("aka_title", "movie_id"): ("title", "id"),
        ("cast_info", "person_id"): ("name", "id"),
        ("cast_info", "movie_id"): ("title", "id"),
        ("cast_info", "person_role_id"): ("char_name", "id"),
        ("cast_info", "role_id"): ("role_type", "id"),
        ("complete_cast", "movie_id"): ("title", "id"),
        ("complete_cast", "subject_id"): ("comp_cast_type", "id"),
        ("complete_cast", "status_id"): ("comp_cast_type", "id"),
        ("movie_companies", "movie_id"): ("title", "id"),
        ("movie_companies", "company_id"): ("company_name", "id"),
        ("movie_companies", "company_type_id"): ("company_type", "id"),
        ("movie_info", "movie_id"): ("title", "id"),
        ("movie_info", "info_type_id"): ("info_type", "id"),
        ("movie_info_idx", "movie_id"): ("title", "id"),
        ("movie_info_idx", "info_type_id"): ("info_type", "id"),
        ("movie_keyword", "movie_id"): ("title", "id"),
        ("movie_keyword", "keyword_id"): ("keyword", "id"),
        ("movie_link", "movie_id"): ("title", "id"),
        ("movie_link", "linked_movie_id"): ("title", "id"),
        ("movie_link", "link_type_id"): ("link_type", "id"),
        ("person_info", "person_id"): ("name", "id"),
        ("person_info", "info_type_id"): ("info_type", "id"),
        ("title", "kind_id"): ("kind_type", "id"),
    }

    MOVIE_ID_COLUMNS = {
        ("title", "id"),
        ("aka_title", "movie_id"),
        ("cast_info", "movie_id"),
        ("complete_cast", "movie_id"),
        ("movie_companies", "movie_id"),
        ("movie_info", "movie_id"),
        ("movie_info_idx", "movie_id"),
        ("movie_keyword", "movie_id"),
        ("movie_link", "movie_id"),
        ("movie_link", "linked_movie_id"),
    }

    def __init__(self):
        super().__init__()
        self.random_seed = 13
        self.schema = {}
        self.table_stats = {}
        self.table_columns = {}
        self.total_title_rows = 1

    def setup(self):
        import json

        random.seed(self.random_seed)
        with open("data/schema.json", "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.table_columns = {t: list(cols.keys()) for t, cols in self.schema.items()}
        self.table_stats = {}
        for table_name in self.schema:
            self.table_stats[table_name] = self._build_table_stats(table_name)
        self.total_title_rows = max(
            1, self.table_stats.get("title", {}).get("row_count", 1)
        )
        if not self.table_stats:
            raise RuntimeError("Setup failed: no table statistics were generated")

    def estimate(self, tables: list, filters: list, joins: list) -> int:
        if not self.table_stats:
            raise RuntimeError("Estimator not initialized. Call setup() first.")
        alias_to_table = {}
        for entry in tables:
            if "name" not in entry or "alias" not in entry:
                raise ValueError(f"Invalid table entry: {entry}")
            table_name = entry["name"]
            alias = entry["alias"]
            if table_name not in self.schema:
                raise ValueError(f"Unknown table referenced: {table_name}")
            if alias in alias_to_table:
                raise ValueError(f"Duplicate alias referenced: {alias}")
            alias_to_table[alias] = table_name

        alias_filters = defaultdict(list)
        for filt in filters:
            self._validate_filter_tree(filt, alias_to_table)
            alias = filt.get("alias")
            if alias is not None:
                alias_filters[alias].append(filt)
            else:
                aliases = self._collect_aliases_from_filter(filt)
                if len(aliases) == 1:
                    alias_filters[next(iter(aliases))].append(filt)

        alias_state = {}
        for alias, table_name in alias_to_table.items():
            tstats = self.table_stats[table_name]
            rows = tstats["row_count"]
            sel = self._estimate_alias_selectivity(
                alias, table_name, alias_filters.get(alias, [])
            )
            est_rows = max(rows * sel, 1.0 if rows > 0 else 0.0)
            movie_cov = self._estimate_movie_coverage(table_name, sel, est_rows)
            alias_state[alias] = {
                "table": table_name,
                "rows": est_rows,
                "sel": sel,
                "movie_cov": movie_cov,
                "ndv": self._initial_alias_ndv(alias, table_name, sel, est_rows),
                "unique_keys": self._initial_unique_keys(alias, table_name, est_rows),
            }

        result = None
        if not joins:
            result = 1.0
            for alias in alias_to_table:
                result *= alias_state[alias]["rows"]
        else:
            result = self._estimate_joins(alias_state, joins, alias_to_table)

        total_max = 1.0
        for alias in alias_to_table:
            total_max *= max(alias_state[alias]["rows"], 1.0)
        if result < 1.0 and total_max >= 1.0:
            result = 1.0
        if result > total_max and total_max > 0:
            result = total_max
        return int(max(1, round(result)))

    def _build_table_stats(self, table_name):
        columns = self.table_columns[table_name]
        rows = []
        reader = self.custom_csv_reader(table_name)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Table {table_name} is empty or unreadable")
        row_count = 0
        target = self.SAMPLE_SIZES.get(table_name, 50000)
        for row in reader:
            row_count += 1
            if len(row) != len(columns):
                continue
            if len(rows) < target:
                rows.append(row)
            else:
                idx = random.randint(1, row_count)
                if idx <= target:
                    rows[idx - 1] = row
        idx_map = {col: i for i, col in enumerate(columns)}
        table_stat = {"row_count": row_count, "columns": {}, "sample_size": len(rows)}
        for col in columns:
            values = [
                self._convert_value(table_name, col, r[idx_map[col]]) for r in rows
            ]
            table_stat["columns"][col] = self._build_column_stat(
                table_name, col, values, rows, idx_map
            )
        if "movie_id" in idx_map:
            movie_vals = [
                self._convert_value(table_name, "movie_id", r[idx_map["movie_id"]])
                for r in rows
            ]
            movie_non_null = [v for v in movie_vals if v is not None]
            movie_counter = Counter(movie_non_null)
            ndv = len(movie_counter)
            avg_rows = (len(movie_non_null) / ndv) if ndv else 1.0
            table_stat["movie_domain"] = {
                "ndv_sample": ndv,
                "avg_rows_per_movie": max(1.0, avg_rows),
                "coverage": min(1.0, ndv / max(1, self.total_title_rows)),
            }
        elif table_name == "title":
            table_stat["movie_domain"] = {
                "ndv_sample": row_count,
                "avg_rows_per_movie": 1.0,
                "coverage": 1.0,
            }
        return table_stat

    def _build_column_stat(self, table, col, values, rows, idx_map):
        non_null = [v for v in values if v is not None]
        null_frac = 0.0 if not values else (len(values) - len(non_null)) / len(values)
        schema_type = self.schema[table][col]
        stat = {
            "null_frac": null_frac,
            "sample_count": len(values),
            "ndv_sample": len(set(non_null)) if non_null else 0,
        }

        if col == "id":
            stat.update({"type": "pk", "is_unique": True})
            return stat

        if (table, col) in self.TEXT_CONFIG:
            cfg = self.TEXT_CONFIG[(table, col)]
            return self._build_text_stat(non_null, null_frac, cfg)

        if (table, col) in self.TOKEN_CONFIG:
            return self._build_token_stat(
                non_null, null_frac, self.TOKEN_CONFIG[(table, col)]
            )

        if table == "name" and col == "name_pcode_cf":
            return self._build_prefix_only_stat(non_null, null_frac, 2)

        if table in self.CONDITIONAL_INFO_TABLES and col == "info":
            return self._build_conditional_info_stat(table, rows, idx_map, null_frac)

        if table == "movie_info_idx" and col == "note":
            stat.update({"type": "null_only"})
            return stat

        if schema_type == "int":
            return self._build_int_stat(table, col, non_null, null_frac)

        return self._build_categorical_stat(non_null, null_frac, 256)

    def _build_int_stat(self, table, col, non_null, null_frac):
        values = [v for v in non_null if isinstance(v, int)]
        counter = Counter(values)
        stat = {"null_frac": null_frac}
        if col in {
            "kind_id",
            "role_id",
            "company_type_id",
            "subject_id",
            "status_id",
            "info_type_id",
            "link_type_id",
            "gender",
        }:
            return self._build_categorical_stat(values, null_frac, 64)
        if col in {"production_year", "episode_nr"}:
            stat.update(
                self._build_hist_stat(
                    values,
                    null_frac,
                    64 if col == "production_year" else 32,
                    128 if col == "episode_nr" else 0,
                )
            )
            return stat
        if (table, col) in self.PK_FK:
            return self._build_fk_stat(
                counter, len(values), null_frac, 8192 if col == "movie_id" else 4096
            )
        return self._build_categorical_stat(values, null_frac, 256)

    def _build_categorical_stat(self, values, null_frac, k):
        counter = Counter(values)
        total = sum(counter.values())
        most = counter.most_common(k)
        top = {v: c / total for v, c in most} if total else {}
        covered = sum(c for _, c in most)
        return {
            "type": "categorical",
            "null_frac": null_frac,
            "freq": top,
            "other_mass": max(0.0, 1.0 - (covered / total if total else 0.0)),
            "other_ndv": max(0, len(counter) - len(top)),
            "ndv_sample": len(counter),
        }

    def _build_hist_stat(self, values, null_frac, bucket_count, topk):
        if not values:
            return {
                "type": "hist",
                "null_frac": null_frac,
                "buckets": [],
                "topk": {},
                "min": None,
                "max": None,
                "ndv_sample": 0,
            }
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        buckets = []
        for i in range(bucket_count):
            start = int(i * n / bucket_count)
            end = int((i + 1) * n / bucket_count)
            if start >= end:
                continue
            lo = sorted_vals[start]
            hi = sorted_vals[end - 1]
            buckets.append((lo, hi, (end - start) / n))
        counter = Counter(values)
        top = {v: c / n for v, c in counter.most_common(topk)} if topk else {}
        return {
            "type": "hist",
            "null_frac": null_frac,
            "buckets": buckets,
            "topk": top,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "ndv_sample": len(counter),
            "total_non_null": n,
        }

    def _build_fk_stat(self, counter, total, null_frac, k):
        most = counter.most_common(k)
        top = {v: c / total for v, c in most} if total else {}
        covered_rows = sum(c for _, c in most)
        return {
            "type": "fk",
            "null_frac": null_frac,
            "topk": top,
            "ndv_sample": len(counter),
            "tail_mass": max(0.0, 1.0 - (covered_rows / total if total else 0.0)),
            "tail_ndv": max(0, len(counter) - len(top)),
        }

    def _build_text_stat(self, values, null_frac, cfg):
        total = len(values)
        counter = Counter(values)
        exact_top = (
            {v: c / total for v, c in counter.most_common(cfg["exact_k"])}
            if total
            else {}
        )
        prefix_freq = {}
        prefix_len = cfg.get("prefix", 0)
        if prefix_len > 0 and total:
            pcount = Counter()
            for v in values:
                s = str(v)
                for l in range(1, prefix_len + 1):
                    if len(s) >= l:
                        pcount[s[:l].lower()] += 1
            prefix_freq = {p: c / total for p, c in pcount.items()}
        tri_counter = Counter()
        for v in values:
            for tri in self._extract_trigrams(str(v).lower()):
                tri_counter[tri] += 1
        trigram_top = (
            {t: c / total for t, c in tri_counter.most_common(cfg["trigram_k"])}
            if total
            else {}
        )
        return {
            "type": "text",
            "null_frac": null_frac,
            "exact": exact_top,
            "prefix": prefix_freq,
            "trigram": trigram_top,
            "ndv_sample": len(counter),
        }

    def _build_token_stat(self, values, null_frac, patterns):
        total = len(values)
        lowered = [str(v).lower() for v in values]
        pat_counts = {}
        for pat in patterns:
            p = pat.lower()
            pat_counts[p] = sum(1 for v in lowered if p in v) / total if total else 0.0
        pairs = {}
        for i, p1 in enumerate(patterns):
            for p2 in patterns[i + 1 :]:
                a = p1.lower()
                b = p2.lower()
                key = tuple(sorted((a, b)))
                pairs[key] = (
                    sum(1 for v in lowered if a in v and b in v) / total
                    if total
                    else 0.0
                )
        exact = Counter(values)
        return {
            "type": "token",
            "null_frac": null_frac,
            "token_df": pat_counts,
            "pair_df": pairs,
            "exact": {v: c / total for v, c in exact.most_common(256)} if total else {},
            "ndv_sample": len(exact),
        }

    def _build_prefix_only_stat(self, values, null_frac, max_len):
        total = len(values)
        counts = Counter()
        for v in values:
            s = str(v)
            for l in range(1, max_len + 1):
                if len(s) >= l:
                    counts[s[:l].lower()] += 1
        return {
            "type": "prefix_only",
            "null_frac": null_frac,
            "prefix": {k: c / total for k, c in counts.items()} if total else {},
            "ndv_sample": len(set(values)),
        }

    def _build_conditional_info_stat(self, table, rows, idx_map, null_frac):
        groups = defaultdict(list)
        type_counter = Counter()
        for row in rows:
            info_t = self._convert_value(
                table, "info_type_id", row[idx_map["info_type_id"]]
            )
            info_v = self._convert_value(table, "info", row[idx_map["info"]])
            if info_t is not None:
                type_counter[info_t] += 1
            if info_v is not None:
                groups[info_t].append(info_v)
        substats = {}
        for key, vals in groups.items():
            if table == "movie_info_idx":
                substats[key] = self._build_lex_or_exact(vals)
            else:
                uniq = len(set(vals))
                lower_vals = {str(v).lower() for v in vals}
                hint_mode = None
                for mode, vocab in self.CONDITIONAL_INFO_HINTS.get(table, {}).items():
                    if lower_vals & vocab:
                        hint_mode = mode
                        break
                if uniq <= 64 or hint_mode in {"genres", "countries"}:
                    substats[key] = self._build_categorical_stat(vals, 0.0, 128)
                else:
                    substats[key] = self._build_text_stat(
                        vals, 0.0, {"prefix": 4, "exact_k": 256, "trigram_k": 2048}
                    )
        return {
            "type": "conditional",
            "null_frac": null_frac,
            "by_type": substats,
            "type_priors": {
                k: v / max(1, sum(type_counter.values()))
                for k, v in type_counter.items()
            },
        }

    def _build_lex_or_exact(self, values):
        uniq = len(set(values))
        if uniq <= 64:
            return self._build_categorical_stat(values, 0.0, 128)
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        buckets = []
        bucket_count = 32
        for i in range(bucket_count):
            start = int(i * n / bucket_count)
            end = int((i + 1) * n / bucket_count)
            if start >= end:
                continue
            buckets.append(
                (sorted_vals[start], sorted_vals[end - 1], (end - start) / n)
            )
        return {
            "type": "lexhist",
            "null_frac": 0.0,
            "buckets": buckets,
            "ndv_sample": uniq,
        }

    def _estimate_alias_selectivity(self, alias, table, filters):
        if not filters:
            return 1.0
        sels = []
        info_type_eq = None
        for filt in filters:
            if (
                filt.get("alias") == alias
                and filt.get("column") == "info_type_id"
                and filt.get("operator") == "="
            ):
                info_type_eq = self._convert_literal(
                    table, "info_type_id", filt.get("literal")
                )
        for filt in filters:
            sels.append(self._estimate_filter_tree(alias, table, filt, info_type_eq))
        sel = 1.0
        for s in sels:
            sel *= s
        if sels:
            sel = max(sel, min(sels) * 0.05)
        return self._clamp_prob(sel)

    def _estimate_filter_tree(self, alias, table, filt, info_type_eq=None):
        if "children" in filt:
            op = filt["operator"]
            child_sels = [
                self._estimate_filter_tree(alias, table, c, info_type_eq)
                for c in filt["children"]
            ]
            if not child_sels:
                return 1.0
            if op == "AND":
                out = 1.0
                for s in child_sels:
                    out *= s
                return self._clamp_prob(max(out, min(child_sels) * 0.05))
            if op == "OR":
                inv = 1.0
                for s in child_sels:
                    inv *= 1.0 - self._clamp_prob(s)
                return self._clamp_prob(1.0 - inv)
            raise ValueError(f"Unsupported logical operator: {op}")
        if filt["alias"] != alias:
            return 1.0
        col = filt["column"]
        op = filt["operator"]
        lit = filt.get("literal")
        stat = self.table_stats[table]["columns"][col]
        if stat["type"] == "conditional":
            return self._estimate_conditional_leaf(
                stat, table, col, op, lit, info_type_eq
            )
        return self._estimate_leaf(stat, table, col, op, lit)

    def _estimate_leaf(self, stat, table, col, op, lit):
        null_frac = stat.get("null_frac", 0.0)
        if op == "IS NULL":
            return self._clamp_prob(null_frac)
        if op == "IS NOT NULL":
            return self._clamp_prob(1.0 - null_frac)
        non_null_mass = max(1e-9, 1.0 - null_frac)
        if op == "!=":
            return self._clamp_prob(
                non_null_mass
                * (
                    1.0
                    - self._estimate_leaf(stat, table, col, "=", lit) / non_null_mass
                )
            )
        if op == "NOT LIKE":
            return self._clamp_prob(
                non_null_mass
                * (
                    1.0
                    - self._estimate_leaf(stat, table, col, "LIKE", lit) / non_null_mass
                )
            )
        if op == "IN":
            vals = lit if isinstance(lit, list) else []
            total = 0.0
            for v in vals:
                total += self._estimate_leaf(stat, table, col, "=", v)
            return self._clamp_prob(total)
        if stat["type"] in {"categorical", "pk"}:
            return self._estimate_categorical(stat, op, lit)
        if stat["type"] == "fk":
            return self._estimate_fk_filter(stat, op, lit)
        if stat["type"] == "hist":
            return self._estimate_hist_filter(stat, op, lit)
        if stat["type"] == "text":
            return self._estimate_text_filter(stat, op, lit)
        if stat["type"] == "token":
            return self._estimate_token_filter(stat, op, lit)
        if stat["type"] == "prefix_only":
            return self._estimate_prefix_only_filter(stat, op, lit)
        if stat["type"] == "lexhist":
            return self._estimate_lex_filter(stat, op, lit)
        if stat["type"] == "null_only":
            return self._clamp_prob(
                null_frac
                if op == "IS NULL"
                else non_null_mass
                if op == "IS NOT NULL"
                else 0.1
            )
        if stat["type"] == "conditional":
            return 0.1
        return 0.1

    def _estimate_categorical(self, stat, op, lit):
        ndv = max(1, stat.get("ndv_sample", 1))
        if op == "=":
            return self._clamp_prob(
                stat["freq"].get(
                    lit,
                    stat.get("other_mass", 0.0) / max(1, stat.get("other_ndv", ndv)),
                )
            )
        return 0.1

    def _estimate_fk_filter(self, stat, op, lit):
        if op == "=":
            return self._clamp_prob(
                stat["topk"].get(lit, stat["tail_mass"] / max(1, stat["tail_ndv"]))
            )
        return 0.1

    def _estimate_hist_filter(self, stat, op, lit):
        if not stat["buckets"]:
            return 0.1
        if op == "=":
            if lit in stat.get("topk", {}):
                return self._clamp_prob(stat["topk"][lit])
            return self._clamp_prob(
                (1.0 - stat["null_frac"]) / max(1, stat.get("ndv_sample", 100))
            )
        minv = stat["min"]
        maxv = stat["max"]
        if minv is None or maxv is None:
            return 0.1
        try:
            v = int(lit)
        except Exception:
            return 0.1
        if maxv == minv:
            if op in (">=", "<=") and v == minv:
                return 1.0 - stat["null_frac"]
            if op == ">" and minv > v:
                return 1.0 - stat["null_frac"]
            if op == "<" and minv < v:
                return 1.0 - stat["null_frac"]
            return 0.0
        total = 0.0
        for lo, hi, mass in stat["buckets"]:
            if hi < lo:
                lo, hi = hi, lo
            width = max(1.0, hi - lo + 1)
            if op == ">":
                if hi <= v:
                    continue
                if lo > v:
                    total += mass
                else:
                    total += mass * max(0.0, (hi - v) / width)
            elif op == ">=":
                if hi < v:
                    continue
                if lo >= v:
                    total += mass
                else:
                    total += mass * max(0.0, (hi - v + 1) / width)
            elif op == "<":
                if lo >= v:
                    continue
                if hi < v:
                    total += mass
                else:
                    total += mass * max(0.0, (v - lo) / width)
            elif op == "<=":
                if lo > v:
                    continue
                if hi <= v:
                    total += mass
                else:
                    total += mass * max(0.0, (v - lo + 1) / width)
        if stat.get("topk"):
            for value, mass in stat["topk"].items():
                if (
                    (op == ">" and value > v)
                    or (op == ">=" and value >= v)
                    or (op == "<" and value < v)
                    or (op == "<=" and value <= v)
                ):
                    total = max(total, mass)
        return self._clamp_prob(total)

    def _estimate_text_filter(self, stat, op, lit):
        if op == "=":
            ndv = max(1, stat.get("ndv_sample", 1))
            return self._clamp_prob(
                stat["exact"].get(lit, (1.0 - stat["null_frac"]) / ndv)
            )
        if op == "LIKE":
            if not isinstance(lit, str):
                return 0.1
            pattern = lit
            if "%" not in pattern and "_" not in pattern:
                return self._estimate_text_filter(stat, "=", lit)
            if (
                pattern.endswith("%")
                and not pattern.startswith("%")
                and pattern.count("%") == 1
                and "_" not in pattern
            ):
                prefix = pattern[:-1].lower()
                return self._clamp_prob(
                    stat["prefix"].get(
                        prefix,
                        stat["prefix"].get(prefix[:1], 0.05)
                        * (0.2 ** max(0, len(prefix) - 1)),
                    )
                )
            chunks = [c.lower() for c in re.split(r"[%_]+", pattern) if c]
            if not chunks:
                return 1.0 - stat["null_frac"]
            sels = []
            for chunk in chunks:
                if len(chunk) >= 3:
                    tris = self._extract_trigrams(chunk)
                    tri_sel = min(
                        [stat["trigram"].get(t, 0.05) for t in tris] or [0.05]
                    )
                    sels.append(tri_sel)
                else:
                    sels.append(stat["prefix"].get(chunk, 0.2))
            out = min(sels)
            for s in sels[1:]:
                out *= max(min(1.0, s * 0.5), 0.01)
            return self._clamp_prob(out)
        return 0.1

    def _estimate_token_filter(self, stat, op, lit):
        if op == "=":
            ndv = max(1, stat.get("ndv_sample", 1))
            return self._clamp_prob(
                stat["exact"].get(lit, (1.0 - stat["null_frac"]) / ndv)
            )
        if op == "LIKE":
            pats = [c.lower() for c in re.split(r"[%_]+", str(lit)) if c]
            if not pats:
                return 1.0 - stat["null_frac"]
            vals = []
            for p in pats:
                vals.append(stat["token_df"].get(p, 0.05))
            if len(vals) == 2:
                key = tuple(sorted((pats[0], pats[1])))
                pair = stat["pair_df"].get(key)
                if pair is not None and pair > 0:
                    return self._clamp_prob(pair)
            out = min(vals)
            for s in vals[1:]:
                out *= max(s, 0.5)
            return self._clamp_prob(out)
        return 0.1

    def _estimate_prefix_only_filter(self, stat, op, lit):
        if op == "LIKE":
            pattern = str(lit)
            if pattern.endswith("%") and not pattern.startswith("%"):
                prefix = pattern[:-1].lower()
                return self._clamp_prob(stat["prefix"].get(prefix, 0.05))
        if op in {">", ">=", "<", "<="}:
            if not isinstance(lit, str):
                lit = str(lit)
            c = lit[:1].lower()
            mass = sum(
                v
                for k, v in stat["prefix"].items()
                if len(k) == 1
                and ((op in {">", ">="} and k >= c) or (op in {"<", "<="} and k <= c))
            )
            return self._clamp_prob(mass)
        return 0.1

    def _estimate_conditional_leaf(self, stat, table, col, op, lit, info_type_eq):
        if info_type_eq is not None:
            inner = stat["by_type"].get(info_type_eq)
            if inner is None:
                return 0.01 if op in {"=", "IN", "LIKE"} else 0.1
            return self._estimate_leaf(inner, table, col, op, lit)

        priors = stat.get("type_priors", {})
        if not priors:
            return 0.1
        if op == "IN" and isinstance(lit, list):
            return self._clamp_prob(
                sum(
                    self._estimate_conditional_leaf(
                        stat, table, col, "=", v, info_type_eq
                    )
                    for v in lit
                )
            )
        total = 0.0
        for type_id, prior in priors.items():
            inner = stat["by_type"].get(type_id)
            if inner is None:
                continue
            total += prior * self._estimate_leaf(inner, table, col, op, lit)
        return self._clamp_prob(total if total > 0 else 0.1)

    def _estimate_lex_filter(self, stat, op, lit):
        buckets = stat.get("buckets", [])
        if not buckets:
            return 0.1
        if op == "=":
            return self._clamp_prob(1.0 / max(1, stat.get("ndv_sample", 100)))
        total = 0.0
        for lo, hi, mass in buckets:
            if op in {">", ">="} and hi >= lit:
                total += mass
            elif op in {"<", "<="} and lo <= lit:
                total += mass
        return self._clamp_prob(total)

    def _estimate_movie_coverage(self, table, sel, rows):
        tstat = self.table_stats[table]
        if table == "title":
            total_movies = max(1, tstat["row_count"])
            movie_count = min(rows, total_movies)
            return {"fraction": movie_count / total_movies, "count": movie_count}
        if "movie_id" in tstat["columns"]:
            domain = tstat.get("movie_domain", {})
            ndv = max(1, domain.get("ndv_sample", tstat["sample_size"]))
            avg_per_movie = max(1.0, domain.get("avg_rows_per_movie", 1.0))
            movie_count = min(rows / avg_per_movie, rows)
            total_movies = max(self.total_title_rows, ndv, movie_count)
            return {
                "fraction": min(1.0, movie_count / total_movies),
                "count": movie_count,
            }
        if table == "movie_link":
            return {"fraction": min(1.0, sel), "count": rows}
        return None

    def _estimate_joins(self, alias_state, joins, alias_to_table):
        remaining = []
        for join in joins:
            for key in ("alias1", "column1", "alias2", "column2"):
                if key not in join:
                    raise ValueError(f"Invalid join specification: {join}")
            a1, c1, a2, c2 = (
                join["alias1"],
                join["column1"],
                join["alias2"],
                join["column2"],
            )
            if a1 not in alias_to_table or a2 not in alias_to_table:
                raise ValueError(f"Join references unknown alias: {join}")
            t1, t2 = alias_to_table[a1], alias_to_table[a2]
            if c1 not in self.schema[t1] or c2 not in self.schema[t2]:
                raise ValueError(f"Join references unknown column: {join}")
            remaining.append(join)

        components = {alias: alias for alias in alias_state}
        state_map = {
            alias: self._copy_state(alias_state[alias]) for alias in alias_state
        }

        while remaining:
            best_idx = None
            best_rows = None
            best_merge = None
            for idx, join in enumerate(remaining):
                a1, a2 = join["alias1"], join["alias2"]
                c1 = components[a1]
                c2 = components[a2]
                if c1 == c2:
                    best_idx = idx
                    best_rows = state_map[c1]["rows"]
                    best_merge = (c1, c2, None)
                    break
                candidate = self._join_component_states(
                    state_map[c1],
                    state_map[c2],
                    alias_to_table[a1],
                    join["column1"],
                    alias_to_table[a2],
                    join["column2"],
                    a1,
                    a2,
                )
                cand_rows = candidate["rows"]
                score = cand_rows * self._join_priority_factor(join, alias_to_table)
                if best_rows is None or score < best_rows:
                    best_rows = score
                    best_idx = idx
                    best_merge = (c1, c2, candidate)
            join = remaining.pop(best_idx)
            c1, c2, candidate = best_merge
            if c1 == c2:
                continue
            new_name = c1
            state_map[new_name] = candidate
            del state_map[c2]
            for alias, comp in list(components.items()):
                if comp == c2:
                    components[alias] = new_name

        final_components = set(components.values())
        result = 1.0
        for comp in final_components:
            result *= state_map[comp]["rows"]
        return result

    def _is_movie_join_key(self, table, column):
        return (table, column) in self.MOVIE_ID_COLUMNS or (
            table == "title" and column == "id"
        )

    def _estimate_single_join(self, a1, c1, a2, c2, alias_state, alias_to_table):
        t1, t2 = alias_to_table[a1], alias_to_table[a2]
        s1 = alias_state[a1]
        s2 = alias_state[a2]
        rows1, rows2 = s1["rows"], s2["rows"]
        if (
            ((t1, c1) in self.MOVIE_ID_COLUMNS and (t2, c2) in self.MOVIE_ID_COLUMNS)
            or ((t1, c1) in self.MOVIE_ID_COLUMNS and t2 == "title" and c2 == "id")
            or ((t2, c2) in self.MOVIE_ID_COLUMNS and t1 == "title" and c1 == "id")
        ):
            cov1 = s1.get("movie_cov") or {"fraction": 1.0, "count": rows1}
            cov2 = s2.get("movie_cov") or {"fraction": 1.0, "count": rows2}
            common_movies = min(
                cov1["count"],
                cov2["count"],
                max(cov1["fraction"], 1e-9)
                * max(cov2["fraction"], 1e-9)
                * max(cov1["count"], cov2["count"]),
            )
            avg1 = max(1.0, rows1 / max(cov1["count"], 1.0))
            avg2 = max(1.0, rows2 / max(cov2["count"], 1.0))
            return max(1.0, common_movies * avg1 * avg2)
        if self.PK_FK.get((t1, c1)) == (t2, c2):
            return rows1 * self._dimension_key_fraction(t2, a2, alias_state)
        if self.PK_FK.get((t2, c2)) == (t1, c1):
            return rows2 * self._dimension_key_fraction(t1, a1, alias_state)
        ndv1 = max(1, self.table_stats[t1]["columns"][c1].get("ndv_sample", int(rows1)))
        ndv2 = max(1, self.table_stats[t2]["columns"][c2].get("ndv_sample", int(rows2)))
        return max(1.0, rows1 * rows2 / max(ndv1, ndv2))

    def _dimension_key_fraction(self, table, alias, alias_state):
        tstat = self.table_stats[table]
        base = max(1.0, tstat["row_count"])
        return self._clamp_prob(alias_state[alias]["rows"] / base)

    def _initial_alias_ndv(self, alias, table, sel, rows):
        ndv = {}
        for col, stat in self.table_stats[table]["columns"].items():
            if (
                col == "id"
                or (table, col) in self.PK_FK
                or col
                in {
                    "movie_id",
                    "person_id",
                    "company_id",
                    "keyword_id",
                    "linked_movie_id",
                    "role_id",
                    "person_role_id",
                    "info_type_id",
                    "kind_id",
                }
            ):
                base_ndv = stat.get(
                    "ndv_sample",
                    min(
                        self.table_stats[table]["row_count"],
                        self.table_stats[table]["sample_size"],
                    ),
                )
                if col == "id":
                    est = min(rows, self.table_stats[table]["row_count"] * sel)
                else:
                    est = min(rows, base_ndv * math.sqrt(max(sel, 1e-9)))
                ndv[(alias, col)] = max(1.0, est)
        return ndv

    def _initial_unique_keys(self, alias, table, rows):
        unique = set()
        if "id" in self.schema[table]:
            unique.add((alias, "id"))
        return unique

    def _copy_state(self, state):
        return {
            "rows": state["rows"],
            "sel": state["sel"],
            "movie_cov": state.get("movie_cov"),
            "ndv": dict(state.get("ndv", {})),
            "unique_keys": set(state.get("unique_keys", set())),
        }

    def _join_component_states(
        self,
        left,
        right,
        left_table,
        left_col,
        right_table,
        right_col,
        left_alias,
        right_alias,
    ):
        left_key = (left_alias, left_col)
        right_key = (right_alias, right_col)
        left_rows = max(1.0, left["rows"])
        right_rows = max(1.0, right["rows"])
        ndv_left = max(
            1.0,
            left["ndv"].get(
                left_key,
                min(
                    left_rows,
                    self.table_stats[left_table]["columns"][left_col].get(
                        "ndv_sample", left_rows
                    ),
                ),
            ),
        )
        ndv_right = max(
            1.0,
            right["ndv"].get(
                right_key,
                min(
                    right_rows,
                    self.table_stats[right_table]["columns"][right_col].get(
                        "ndv_sample", right_rows
                    ),
                ),
            ),
        )

        left_unique = left_key in left["unique_keys"]
        right_unique = right_key in right["unique_keys"]

        if self._is_aka_name_cast_info_person_join(
            left_table, left_col, right_table, right_col
        ):
            common_persons = min(ndv_left, ndv_right)
            left_fanout = max(1.0, left_rows / max(ndv_left, 1.0))
            right_fanout = max(1.0, right_rows / max(ndv_right, 1.0))
            join_rows = common_persons * left_fanout * right_fanout
            join_rows = min(
                join_rows, left_rows * right_fanout, right_rows * left_fanout
            )
        elif (
            self.PK_FK.get((left_table, left_col)) == (right_table, right_col)
            and right_unique
        ):
            base_pk_ndv = max(1.0, self.table_stats[right_table]["row_count"])
            match_rate = min(1.0, ndv_right / base_pk_ndv)
            join_rows = min(left_rows, left_rows * match_rate)
        elif (
            self.PK_FK.get((right_table, right_col)) == (left_table, left_col)
            and left_unique
        ):
            base_pk_ndv = max(1.0, self.table_stats[left_table]["row_count"])
            match_rate = min(1.0, ndv_left / base_pk_ndv)
            join_rows = min(right_rows, right_rows * match_rate)
        else:
            join_rows = (left_rows * right_rows) / max(ndv_left, ndv_right, 1.0)
            avg_left = left_rows / max(ndv_left, 1.0)
            avg_right = right_rows / max(ndv_right, 1.0)
            join_rows = min(
                join_rows,
                ndv_left * max(avg_right, 1.0),
                ndv_right * max(avg_left, 1.0),
            )
            if left_unique and not right_unique:
                join_rows = min(join_rows, right_rows)
            if right_unique and not left_unique:
                join_rows = min(join_rows, left_rows)
            if (
                self._is_movie_join_key(left_table, left_col)
                and self._is_movie_join_key(right_table, right_col)
                and not left_unique
                and not right_unique
            ):
                movie_ndv = min(ndv_left, ndv_right, float(self.total_title_rows))
                left_fanout = max(1.0, left_rows / max(ndv_left, 1.0))
                right_fanout = max(1.0, right_rows / max(ndv_right, 1.0))
                join_rows = max(join_rows, movie_ndv * left_fanout * right_fanout)
                left_movie_cov = left.get("movie_cov")
                right_movie_cov = right.get("movie_cov")
                if left_movie_cov and right_movie_cov:
                    overlap_movies = min(
                        left_movie_cov["count"], right_movie_cov["count"], movie_ndv
                    )
                    join_rows = max(
                        join_rows, overlap_movies * left_fanout * right_fanout
                    )
        join_rows = max(1.0, join_rows)

        out = {
            "rows": join_rows,
            "sel": min(left.get("sel", 1.0), right.get("sel", 1.0)),
            "movie_cov": None,
            "ndv": {},
            "unique_keys": set(),
        }

        for key, val in left["ndv"].items():
            out["ndv"][key] = min(val, join_rows)
        for key, val in right["ndv"].items():
            out["ndv"][key] = min(out["ndv"].get(key, val), val, join_rows)
        out["ndv"][left_key] = min(ndv_left, ndv_right, join_rows)
        out["ndv"][right_key] = out["ndv"][left_key]

        if left_unique and right_unique:
            out["unique_keys"].add(left_key)
            out["unique_keys"].add(right_key)
        elif left_unique and right_key in right["unique_keys"]:
            out["unique_keys"].add(left_key)
        elif right_unique and left_key in left["unique_keys"]:
            out["unique_keys"].add(right_key)

        for key in left["unique_keys"]:
            if key != left_key and right_unique:
                out["unique_keys"].add(key)
        for key in right["unique_keys"]:
            if key != right_key and left_unique:
                out["unique_keys"].add(key)

        if (
            left.get("movie_cov")
            and right.get("movie_cov")
            and self._is_movie_join_key(left_table, left_col)
            and self._is_movie_join_key(right_table, right_col)
        ):
            common = min(
                left["movie_cov"]["count"],
                right["movie_cov"]["count"],
                out["ndv"][left_key],
            )
            out["movie_cov"] = {
                "count": common,
                "fraction": min(1.0, common / max(1.0, self.total_title_rows)),
            }
        else:
            out["movie_cov"] = left.get("movie_cov") or right.get("movie_cov")
        return out

    def _join_priority_factor(self, join, alias_to_table):
        a1, c1, a2, c2 = (
            join["alias1"],
            join["column1"],
            join["alias2"],
            join["column2"],
        )
        t1, t2 = alias_to_table[a1], alias_to_table[a2]
        if (
            t1 == "name"
            and c1 == "id"
            and t2 in {"cast_info", "aka_name"}
            and c2 == "person_id"
        ) or (
            t2 == "name"
            and c2 == "id"
            and t1 in {"cast_info", "aka_name"}
            and c1 == "person_id"
        ):
            return 0.5
        return 1.0

    def _is_aka_name_cast_info_person_join(
        self, left_table, left_col, right_table, right_col
    ):
        return (
            left_col == "person_id"
            and right_col == "person_id"
            and {left_table, right_table} == {"aka_name", "cast_info"}
        )

    def _validate_filter_tree(self, filt, alias_to_table):
        if "children" in filt:
            if filt.get("operator") not in {"AND", "OR"}:
                raise ValueError(f"Invalid logical operator: {filt}")
            for child in filt["children"]:
                self._validate_filter_tree(child, alias_to_table)
            return
        alias = filt.get("alias")
        col = filt.get("column")
        if alias not in alias_to_table:
            raise ValueError(f"Unknown alias in filter: {alias}")
        table = alias_to_table[alias]
        if col not in self.schema[table]:
            raise ValueError(f"Unknown column in filter: {table}.{col}")

    def _collect_aliases_from_filter(self, filt):
        if "children" in filt:
            out = set()
            for child in filt["children"]:
                out |= self._collect_aliases_from_filter(child)
            return out
        return {filt["alias"]}

    def _convert_value(self, table, col, raw):
        if raw == "" or raw is None:
            return None
        typ = self.schema[table][col]
        if typ == "int":
            try:
                return int(raw)
            except Exception:
                return None
        return raw

    def _convert_literal(self, table, col, lit):
        if lit is None:
            return None
        if self.schema[table][col] == "int":
            try:
                return int(lit)
            except Exception:
                return lit
        return lit

    def _extract_trigrams(self, text):
        if len(text) < 3:
            return [text] if text else []
        return [text[i : i + 3] for i in range(len(text) - 2)]

    def _clamp_prob(self, p):
        return max(0.0, min(1.0, p))
