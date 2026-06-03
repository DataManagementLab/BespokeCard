import json
import os
import time
from collections import defaultdict
import random
from typing import Any, Dict, List, Set, Tuple

import psycopg2
from duckdb import cursor
from tqdm import tqdm
import duckdb

import logging

logger = logging.getLogger(__name__)


class QueryExecutor:
    def __init__(self):
        self.connection = psycopg2.connect(database="imdb", host="localhost", port=5432)
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()
        self.cursor.execute("set statement_timeout to 150000;")

    def execute(self, query: str):
        """Executes any SQL statement, returns rows only for SELECT."""
        try:
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except psycopg2.Error as e:
            print(f"Query execution timed out.")
            return None


def run_duckdb_query(sql: str) -> Dict[str, Any]:
    """
    Run a SQL query against the database using DuckDB.

    Args:
        sql: A string containing the SQL query to execute.

    Returns:
        A dictionary containing:
            columns: A list of column names returned by the query.
            rows: A list of tuples representing the rows returned by the query.
            row_count: The number of rows returned by the query.
    """
    DB_PATH = "data/imdb.duckdb"
    con = duckdb.connect(DB_PATH, read_only=True)
    # check if query has 'at' in it - if yes, we replace it with 'at1' since 'at' is reserved in duckdb
    if "AS at" in sql or "at." in sql:
        sql = sql.replace("AS at", "AS at1")
        sql = sql.replace("at.", "at1.")

    try:
        cursor = con.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }
    finally:
        con.close()


def extract_alias_dict(sql: str) -> Dict[str, str]:
    # extract tables from the SQL query
    assert "FROM" in sql, "FROM clause not found in the SQL query"
    assert "WHERE" in sql, "WHERE clause not found in the SQL query"
    from_clause = sql.split("FROM")[1].split("WHERE")[0].strip()

    # extract the table names from the from clause
    if " JOIN " in from_clause:
        items = from_clause.split(" ")
    else:
        items = from_clause.split(",")

    # check for table name aliases
    alias_dict = dict()
    for i in items:
        if " AS " in i:
            table, alias = i.split(" AS ")
            alias_dict[alias.strip()] = table.strip()
        elif " " in i.strip():
            try:
                table, alias = i.strip().split(" ")
            except ValueError as e:
                print(f"Error parsing table name: {i.strip()}")
                raise e
            alias_dict[alias.strip()] = table.strip()
        else:
            alias_dict[i.strip()] = i.strip()

    return alias_dict


def _find_cols_connected_to_this_col(
    table_name: str,
    col_name: str,
    join_by_table_dict: Dict[str, Dict[str, List[Tuple[str, str]]]],
) -> Set[Tuple[str, str]]:
    # connected cols
    connected_cols = set()

    # add starting node in join graph
    connected_cols.add((table_name, col_name))

    # task queue
    task_queue = [(table_name, col_name)]

    while len(task_queue) > 0:
        # pop first item
        table_name, col_name = task_queue.pop(0)

        # check if there is a join on this column
        if col_name in join_by_table_dict[table_name]:
            # get all join partners
            join_partners = join_by_table_dict[table_name][col_name]

            for partner in join_partners:
                # check if partner is already in connected cols
                if partner not in connected_cols:
                    # add to connected cols
                    connected_cols.add(partner)

                    # add to task queue
                    task_queue.append(partner)

    return connected_cols


def _augment_transitive_join_conds(
    join_conds: Dict[Tuple[str, str], List[str]],
) -> Dict[Tuple[str, str], List[str]]:
    # create join dict by table
    join_by_table_dict = defaultdict(dict)  # table -> column -> [(table, column)]

    for table_pair, join_conditions in join_conds.items():
        assert len(table_pair) == 2, (
            f"Join condition {table_pair} does not have 2 tables: {join_conds}"
        )

        for jc in join_conditions:
            # parse join condition
            assert " = " in jc, f"Join condition {jc} does not contain ="

            left, right = jc.split(" = ")

            assert "." in left, f"Left side of join condition {jc} does not contain ."
            assert "." in right, f"Right side of join condition {jc} does not contain ."

            table1, column1 = left.split(".")
            table2, column2 = right.split(".")

            if column1 not in join_by_table_dict[table1]:
                join_by_table_dict[table1][column1] = []
            if column2 not in join_by_table_dict[table2]:
                join_by_table_dict[table2][column2] = []

            join_by_table_dict[table1][column1].append((table2, column2))
            join_by_table_dict[table2][column2].append((table1, column1))

    for table_name, items in join_by_table_dict.items():
        for column_name, join_partners in items.items():
            connected_cols = _find_cols_connected_to_this_col(
                table_name, column_name, join_by_table_dict=join_by_table_dict
            )

            if len(connected_cols) > 2:
                # we have a transitive join condition - e.g. a.col = b.col AND b.col = c.col (or even a full cycle)
                # augment join conditions

                for partner_table, partner_col in connected_cols:
                    if partner_table == table_name and partner_col == column_name:
                        # skip
                        continue

                    # add join condition
                    join_str = " = ".join(
                        sorted(
                            [
                                f"{table_name}.{column_name}",
                                f"{partner_table}.{partner_col}",
                            ]
                        )
                    )
                    reversely_ordered_join_str = " = ".join(
                        sorted(
                            [
                                f"{partner_table}.{partner_col}",
                                f"{table_name}.{column_name}",
                            ],
                            reverse=True,
                        )
                    )

                    tables_tuples: Tuple[str, str] = tuple(
                        sorted([table_name, partner_table])
                    )

                    # check that this join condition has not been seen before
                    if (
                        join_str not in join_conds[tables_tuples]
                        and reversely_ordered_join_str not in join_conds[tables_tuples]
                    ):
                        join_conds[tables_tuples].append(join_str)

    return join_conds


def _extract_filter_and_join_conds_from_sql(
    sql: str,
) -> Tuple[
    List[str], Dict[str, List[str]], Dict[Tuple[str, str], List[str]], Dict[str, str]
]:
    # remove trailing ;
    if sql.endswith(";"):
        sql = sql[:-1]

    alias_dict = extract_alias_dict(sql=sql)

    table_aliases = list(alias_dict.keys())
    table_aliases = sorted(table_aliases)

    # get all available tables in the dataset
    with open("data/schema.json", "r") as f:
        schema = json.load(f)
    # schema = #get_json_schema(dataset)
    tables = list(schema.keys())

    # check that all table found are in the schema
    for t in table_aliases:
        if alias_dict[t] not in tables:
            # try it with lower case?
            alias_dict[t] = alias_dict[t].lower()

        assert alias_dict[t] in tables, (
            f"Table/Alias {t} not found in the schema: {tables} \n {alias_dict}"
        )

    # extract the filter columns from the where clause
    where_clause = sql.split("WHERE")[1].strip()
    clauses = where_clause.split("AND")

    # merge back BETWEEN clauses
    fixed_clauses = []
    i = 0
    while i < len(clauses):
        clause = clauses[i]
        if " BETWEEN " in clauses[i]:
            merged_clause = f"{clauses[i]} AND {clauses[i + 1]}"
            fixed_clauses.append(merged_clause)
            # skip the next clause
            i += 1
        elif clauses[i] in [""" (n.gender='m' OR (n.gender = 'f' """]:
            # manual handling of this clause
            merged_clause = f"{clauses[i]} AND {clauses[i + 1]}"
            fixed_clauses.append(merged_clause.strip())
            # skip the next clause
            i += 1
        else:
            fixed_clauses.append(clauses[i])
        i += 1

    clauses = fixed_clauses

    # remove newlines and tabs
    clauses = [c.replace("\n", " ").replace("\t", " ") for c in clauses]

    # parse the filters
    filters = defaultdict(list)
    join_cond = defaultdict(list)

    for clause in clauses:
        clause = clause.strip()
        tables_found = []

        if clause in [
            """(n.gender='m' OR (n.gender = 'f'  AND  n.name LIKE 'A%'))""",
            """(n.gender='m' OR (n.gender = 'f'  AND  n.name LIKE 'B%'))""",
        ]:
            # manual handling of this clause
            tables_found.append("n")
        else:
            if clause.startswith("(") and clause.endswith(")") and " OR " not in clause:
                # strip brackets
                clause = clause[1:-1]

            # support for (a.col=1 OR a.col=2) clause
            if clause.startswith("(") and clause.endswith(")"):
                assert " OR " in clause, f"OR not found in clause: {clause}"
                or_clauses = clause[1:-1].split(" OR ")
                or_tables = []

                for or_clause in or_clauses:
                    for t in table_aliases:
                        if or_clause.strip().startswith(f"{t}."):
                            or_tables.append(t)
                assert len(or_tables) == len(or_clauses), (
                    f"Could not find all tables in the OR clause: {clause}, tables: {table_aliases}"
                )
                or_tables = set(or_tables)

                assert len(or_tables) == 1, (
                    f"More than one table found in the OR clause: {clause}, tables: {table_aliases}"
                )
                tables_found.append(list(or_tables)[0])
            else:
                assert " OR " not in clause, f"OR found in clause: {clause}"
                for t in table_aliases:
                    if clause.strip().startswith(f"{t}."):
                        tables_found.append(t)
                    elif f"= {t}." in clause:
                        tables_found.append(t)
                    elif f"={t}." in clause:
                        tables_found.append(t)

        # sort tables alphabetically
        tables_found = sorted(set(tables_found))

        if len(tables_found) == 1:
            filters[tables_found[0]].append(clause.strip())
        elif len(tables_found) == 2:
            join_cond[tuple(tables_found)].append(clause.strip())
        elif len(tables_found) > 2:
            raise Exception(
                f"More than 2 tables found in the clause: {clause}, tables: {table_aliases}"
            )
        else:
            raise Exception(
                f"Could not find the table names in the clause: {clause}, tables: {table_aliases}"
            )

    return table_aliases, filters, join_cond, alias_dict


def _compute_join_steps_dict(
    join_conds: Dict[Tuple[str, str], List[str]], table_aliases: List[str]
) -> Dict[int, Set[Tuple[str]]]:
    # compute all join partners for each table
    join_partners = defaultdict(list)
    for join_cond in join_conds:
        join_partners[join_cond[0]].append(join_cond[1])
        join_partners[join_cond[1]].append(join_cond[0])

    join_steps_dict = defaultdict(set)
    join_steps_dict[0] = set(tuple([t]) for t in table_aliases)

    for i in range(1, len(table_aliases)):
        prev_joins = join_steps_dict[i - 1]

        for join_combination in prev_joins:
            for table in join_combination:
                # retrieve all possible join partners of this table
                for join_partner in join_partners[table]:
                    if join_partner not in join_combination:
                        new_join_combination = tuple(
                            sorted(list(join_combination) + [join_partner])
                        )
                        join_steps_dict[i].add(new_join_combination)

    return join_steps_dict


def transform_filter(filter: str):
    # nested
    # if " OR " in filter or (" AND " in filter and not "BETWEEN" in filter):
    #    print(filter)
    if filter.startswith("(") and filter.endswith(")"):
        filter = filter[1:-1].strip()
        if filter.startswith("("):
            print(
                "Warning: first child is nested as well - this is currently not supported"
            )
        # check if either AND or OR occurs first
        and_index = filter.find(" AND ")
        or_index = filter.find(" OR ")
        if and_index != -1 and (or_index == -1 or and_index < or_index):
            operator = "AND"
        elif or_index != -1 and (and_index == -1 or or_index < and_index):
            operator = "OR"
        else:
            raise Exception(f"Could not determine operator in filter: {filter}")
        return {
            "operator": operator,
            "children": [
                transform_filter(child.strip()) for child in filter.split(operator)
            ],
        }
    else:
        operator_types = [
            "<=",
            ">=",
            "!=",
            "<",
            ">",
            "=",
            " NOT LIKE ",
            " IS NULL",
            " IS NOT NULL",
            " IN ",
            " BETWEEN ",
            " LIKE ",
        ]
        for operator in operator_types:
            if operator in filter:
                break
        if len(filter.split(operator)) != 2:
            raise Exception(f"Could not parse filter: {filter}")
        left, right = filter.split(operator)
        alias = left.split(".")[0].strip()
        column = left.split(".")[1].strip()
        operator = operator.strip()
        literal = right.strip()
        if operator in ["IS NULL", "IS NOT NULL"]:
            literal = None
        elif operator == "IN":
            literal = right.strip()
            assert literal.startswith("(") and literal.endswith(")"), (
                f"IN literal not in brackets: {filter}"
            )
            literal = literal[1:-1].split(",")
            literal = [l.strip().strip("'") for l in literal]
        elif operator == "BETWEEN":
            literal = right.strip()
            assert " AND " in literal, f"BETWEEN literal does not contain AND: {filter}"
            lower, upper = literal.split(" AND ")
            literal = (lower.strip().strip("'"), upper.strip().strip("'"))
            return {
                "operator": "AND",
                "children": [
                    transform_filter(f"{alias}.{column} >= {literal[0]}"),
                    transform_filter(f"{alias}.{column} <= {literal[1]}"),
                ],
            }
        else:
            literal = literal.strip().strip("'")

        if (
            type(literal) == str and literal.isnumeric()
        ):  # float remains string as it is movie_info_idx.info column, which contains numeric values but should be treated as strings
            literal = int(literal)
        return {
            "alias": alias,
            "column": column,
            "operator": operator,
            "literal": literal,
        }


def generate_subplans():
    if os.path.exists("outputs/job_subplans.json"):
        logger.info("Subplans file already exists.")
        return
    start_time = time.perf_counter()
    non_coherent_join_graph_ctr = 0
    with open(
        "data/orig_queries.sql",
        "r",
    ) as f:
        sql_queries = f.read().splitlines()

    subplan_count = 0
    subplans_dict = dict()
    for sql in tqdm(sql_queries):
        table_aliases, filters, join_conds, table_alias_dict = (
            _extract_filter_and_join_conds_from_sql(sql)
        )

        # augment transitive join conds
        join_conds = _augment_transitive_join_conds(join_conds)

        # compile list of tables from
        table_names = list(table_alias_dict.values())
        table_names = sorted(table_names)

        join_steps_dict = _compute_join_steps_dict(
            join_conds=join_conds, table_aliases=table_aliases
        )

        max_join_graph = max(join_steps_dict.keys())
        if max_join_graph != len(table_aliases) - 1:
            logger.info(
                f"Join graph is not a single graph: {max_join_graph} != {len(table_aliases) - 1}"
            )
            non_coherent_join_graph_ctr += 1
            continue

        # generate queries testing each sub-plan
        subplan_query_dict = dict()
        for key, values in join_steps_dict.items():
            # if key == 0:
            # no need to execute single table queries (no join in it)
            #    continue

            for join_combination in values:
                # retrieve all filters and join conditions for the current join combination
                tmp_filters = []
                tmp_joins = []
                filters_list = []
                joins_list = []

                for table in join_combination:
                    tmp_filters += filters[table]
                    # retrieve all join conditions for the current join combination
                    for table2 in join_combination:
                        if table == table2:
                            continue

                        if (table, table2) in join_conds:
                            tmp_joins.append(join_conds[(table, table2)][0])
                            j1 = join_conds[(table, table2)][0].split("=")[0].strip()
                            j2 = join_conds[(table, table2)][0].split("=")[1].strip()
                            t1 = j1.split(".")[0]
                            c1 = j1.split(".")[1]
                            t2 = j2.split(".")[0]
                            c2 = j2.split(".")[1]
                            joins_list.append(
                                {
                                    "alias1": t1,
                                    "column1": c1,
                                    "alias2": t2,
                                    "column2": c2,
                                }
                            )
                        # we do not need to check the other way, since we iterate over all combinations (n^2)

                assert len(tmp_joins) >= len(join_combination) - 1, (
                    f"Number of join conditions does not match the number of tables in the join combination: {join_combination}, {tmp_joins}, join_conds: {join_conds}"
                )

                # generate SQL query
                table_as_strings = []
                tables_list = []
                for j in join_combination:
                    table_as_strings.append(f"{table_alias_dict[j]} AS {j}")
                    tables_list.append({"name": table_alias_dict[j], "alias": j})

                # sort alphabetically
                table_as_strings = sorted(table_as_strings)
                tmp_joins = sorted(tmp_joins)
                tmp_filters = sorted(tmp_filters)
                filters_list = [transform_filter(f) for f in tmp_filters]

                if len(tmp_joins) + len(tmp_filters) > 0:
                    count_query = f"SELECT COUNT(*) FROM {','.join(table_as_strings)} WHERE {' AND '.join(tmp_joins + tmp_filters)};"
                else:
                    count_query = f"SELECT COUNT(*) FROM {','.join(table_as_strings)};"

                subplan_query_dict[str(join_combination)] = {
                    "count_query": count_query,
                    "query_raw_info": (tables_list, filters_list, joins_list),
                    "true_card": None,
                    "pg_card": None,
                    "bespoke_card": None,
                }
                subplan_count += 1
        subplans_dict[sql] = subplan_query_dict

    with open("outputs/job_subplans.json", "w") as f:
        json.dump(subplans_dict, f, indent=2)
    logger.info(
        f"Total subplans generated: {subplan_count} in {time.perf_counter() - start_time:.2f} seconds"
    )


def annotate_subplans_with_true_cards(overwrite=False):
    with open("outputs/job_subplans.json", "r") as f:
        subplans_dict = json.load(f)
    start_time = time.perf_counter()
    try:
        for sql, subplans in tqdm(subplans_dict.items()):
            for subplan, items in subplans.items():
                if items["true_card"] is not None and not overwrite:
                    continue
                try:
                    items["true_card"] = run_duckdb_query(items["count_query"])["rows"][
                        0
                    ][0]
                except Exception as e:
                    logger.error(
                        f"Error executing query to get true cardinality: {e}, query: {items['count_query']}"
                    )
                    break

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
    finally:
        with open("outputs/job_subplans.json", "w") as f:
            json.dump(subplans_dict, f, indent=2)
        logger.info(
            f"True card progress preserved in outputs/job_subplans.json. Time taken: {time.perf_counter() - start_time:.2f} seconds"
        )


def annotate_subplans_with_pg_cards(overwrite=False):
    with open("outputs/job_subplans.json", "r") as f:
        subplans_dict = json.load(f)
    executor = QueryExecutor()
    start_time = time.perf_counter()
    try:
        for sql, subplans in tqdm(subplans_dict.items()):
            for subplan, items in subplans.items():
                if items["pg_card"] is not None and not overwrite:
                    continue
                try:
                    # adapt count query
                    query = items["count_query"].replace(
                        "SELECT COUNT(*)", "EXPLAIN (FORMAT JSON) SELECT *"
                    )
                    # execute query
                    result = executor.execute(query)
                    if result:
                        # extract pg estimate
                        pg_card = result[0][0][0]["Plan"]["Plan Rows"]
                    else:
                        pg_card = -1
                except Exception as e:
                    logger.error(f"Error executing query: {e}")
                    break

                items["pg_card"] = pg_card
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
    finally:
        with open("outputs/job_subplans.json", "w") as f:
            json.dump(subplans_dict, f, indent=2)
        logger.info(
            f"PG card progress preserved in outputs/job_subplans.json. Time taken: {time.perf_counter() - start_time:.2f} seconds"
        )


def annotate_subplans_with_bespoke_cards(estimator, no_tqdm=False, overwrite=True):
    with open("outputs/job_subplans.json", "r") as f:
        subplans_dict = json.load(f)
    start_time = time.perf_counter()
    try:
        for sql, subplans in tqdm(subplans_dict.items(), disable=no_tqdm):
            for subplan, items in subplans.items():
                if items["bespoke_card"] is not None and not overwrite:
                    continue
                try:
                    items["bespoke_card"] = int(
                        estimator.estimate(*items["query_raw_info"])
                    )
                except Exception as e:
                    logger.error(f"Error executing query: {e}")
                    raise e
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")

    with open("outputs/job_subplans.json", "w") as f:
        json.dump(subplans_dict, f, indent=2)
    logger.info(
        f"Bespoke card progress preserved in outputs/job_subplans.json. Time taken: {time.perf_counter() - start_time:.2f} seconds"
    )
