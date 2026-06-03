import json
import re
import duckdb

DB_PATH = "data/imdb.duckdb"
con = duckdb.connect(DB_PATH, read_only=True)


def act_card_from_flat_plan(flat_plan):
    def flat_plan_to_sql(flat_plan):
        def visit_filter(f):
            if "children" in f:
                tmp = f"({visit_filter(f['children'][0])}"
                for c in f["children"][1:]:
                    tmp += f" {f['operator']} " + visit_filter(c)
                tmp += ")"
            else:
                # in case literal is of type string, add quotes
                tmp = f"{f['alias']}.{f['column']} {f['operator']}"
                if type(f["literal"]) == str:
                    tmp += f"'{f['literal']}'"
                elif f["literal"] != None:
                    tmp += f"{f['literal']}"
            return tmp

        sql = f"select count(*) from"
        for t in flat_plan["tables"]:
            sql += f" {t['name']} as {t['alias']},"
        sql = sql.strip(",") + " where"
        for f in flat_plan["filters"]:
            sql += f" {visit_filter(f)} and"
        for j in flat_plan["joins"]:
            sql += f" {j['alias1']}.{j['column1']} = {j['alias2']}.{j['column2']} and"
        sql = sql.strip("and").strip("where")

        # change 'at' alias to 'at1' since 'at' is reserved in duckdb
        sql = re.sub(
            r"\bas at\b|\bat\.",
            lambda m: "as at1" if m.group(0) == "as at" else "at1.",
            sql,
        )
        return sql + ";"

    sql = flat_plan_to_sql(flat_plan)
    cursor = con.execute(sql)
    rows = cursor.fetchall()
    card = rows[0][0]
    return sql, card


def filter_default_queries(
    input_path: str = "data/job_w_optimal_w_idxs.json",
    output_path: str = "data/job_default.json",
):
    """From all provided queries, filter those without hints (i.e. postgres default)."""
    with open(input_path, "r") as f:
        queries = json.load(f)

    default_queries = [q for q in queries["parsed_plans"] if q["hint"] == ""]

    with open(output_path, "w") as f:
        json.dump(default_queries, f)


## per plan, we want a flat plan, i.e., a list of tables, filters, joins. Also the SQL query and the true cardinality, PG cardinality
def transform_plan(plan: dict, root=True) -> dict:
    """Transforms nested plan into dictionary containing SQL query, true cardinality, PG cardinality and flattened plan."""

    # start from root's first child, since root has the MIN aggregate
    if root:
        node = plan["children"][0]
    else:
        node = plan
        if node["plan_parameters"]["op_name"] == "Materialize":
            node = node["children"][0]

    transformed_plan = {
        "true_card": node["plan_parameters"]["act_card"],
        "pg_card": node["plan_parameters"]["est_card"],
        "indexes": False,
    }

    if root:
        # strip projections from sql queries
        sql = plan["sql"]
        sql = "SELECT * " + sql[sql.index("FROM") :]
        # sql = "select * " + sql[sql.lower().index("from") :]
        transformed_plan["sql"] = sql
        transformed_plan["subplans"] = []

    tables = []
    filters = []
    joins = []

    def visit_filter(filter: dict):
        if filter["operator"] in ["AND", "OR"]:  # check if filter is nested
            return {
                "operator": filter["operator"],
                "children": [visit_filter(child) for child in filter["children"]],
            }
        else:
            if type(filter["literal"]) == str:
                filter["literal"] = filter["literal"].replace("'", "")
                if filter["operator"] == "IN":
                    filter["literal"] = filter["literal"][1:-1]
                    filter["literal"] = filter["literal"].split(",")
                    filter["literal"] = [
                        literal.strip('"') for literal in filter["literal"]
                    ]
            if (
                type(filter["literal"]) == str
                and filter["literal"].replace(".", "", 1).isnumeric()
            ):
                try:
                    filter["literal"] = int(filter["literal"])
                except:
                    filter["literal"] = float(filter["literal"])
                if (
                    filter["col_name"] == "info" and "mi_idx" in filter["table_alias"]
                ):  # edge case for movie_info_idx.info column, which contains numeric values but should be treated as strings
                    if filter["operator"] == "<=":
                        filter["operator"] = "<"
                        filter["literal"] = filter["literal"] + 1.0
                    elif filter["operator"] == ">=":
                        filter["operator"] = ">"
                        filter["literal"] = filter["literal"] - 1.0
                    filter["literal"] = str(filter["literal"])

            return {
                "alias": filter["table_alias"],
                "column": filter["col_name"],
                "operator": filter["operator"],
                "literal": filter["literal"],
            }

    def visit(node: dict):
        params = node["plan_parameters"]

        if params.keys().__contains__("join"):  # check for join
            if root:
                for child in node["children"]:
                    transformed_plan["subplans"].append(
                        transform_plan(child, root=False)
                    )
            j = params["join"]
            joins.append(
                {
                    "alias1": j["table_alias1"],
                    "column1": j["column_name1"],
                    "alias2": j["table_alias2"],
                    "column2": j["column_name2"],
                }
                # f"{j["table_alias1"]}.{j["column_name1"]} {j["operator"]} {j["table_alias2"]}.{j["column_name2"]}"
            )

        elif params.keys().__contains__("filter"):  # check for filter
            if params["op_name"] in ["Index Scan", "Index Only Scan"]:
                transformed_plan["indexes"] = True
            f = params["filter"]
            filters.append(visit_filter(f))
            ## add filtered table to tables, so mapping from alias to table exists
            while f["operator"] in ["AND", "OR"]:
                f = f["children"][0]

            tables.append({"name": f["table_name"], "alias": f["table_alias"]})

        else:  # otherwise, it's a table scan
            if params["op_name"] in ["Index Scan", "Index Only Scan"]:
                transformed_plan["indexes"] = True
            if params.keys().__contains__("table_name") and params.keys().__contains__(
                "table_alias"
            ):  # ignore hashes and continue with their children
                tables.append(
                    {"name": params["table_name"], "alias": params["table_alias"]}
                    # f"{params["table_name"]} AS {params["table_alias"]}"
                )

        for child in node["children"]:
            visit(child)

    visit(node)

    transformed_plan["flat_plan"] = {
        "tables": tables,
        "filters": filters,
        "joins": joins,
    }

    transformed_plan["num_joins"] = len(joins)
    sql, act_card = act_card_from_flat_plan(transformed_plan["flat_plan"])
    transformed_plan["act_card"] = act_card
    transformed_plan["q"] = sql
    return transformed_plan


def transform_plans(
    input_path: str = "data/job_default.json",
    output_path: str = "data/job_default_flat.json",
):
    """Transforms all plans in the input file and saves them to the output file."""
    with open(input_path, "r") as f:
        plans = json.load(f)

    flat_plans = [transform_plan(plan) for plan in plans]

    with open(output_path, "w") as f:
        json.dump(flat_plans, f)


filter_default_queries(
    # input_path="data/job_light_subplans.json", output_path="data/job_light_default.json"
)
transform_plans(
    # input_path="data/job_light_default.json",
    # output_path="data/job_light_default_flat.json",
)


con.close()
