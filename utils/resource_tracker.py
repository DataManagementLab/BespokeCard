import json


class ResourceTracker:
    def __init__(self, name):
        self.agent_name = name
        self.requests = []
        self.turns: list[dict] = []
        self.request_type = None

    def add_turn(self, response, response_time, cost, from_cache):
        turn = {
            "cost": round(cost, 6),
            "response_time": round(response_time, 6),
            "from_cache": from_cache,
            "tokens": {
                "input_tokens": response.usage.input_tokens,
                "cached_tokens": response.usage.input_tokens_details.cached_tokens,
                "output_tokens": response.usage.output_tokens,
                "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            },
            "response_type": response.output[0].type,
            "tool": (
                None
                if not (
                    response.output[0].type == "function_call"
                    or response.output[0].type == "apply_patch_call"
                )
                else {
                    "name": (
                        response.output[0].name
                        if response.output[0].type == "function_call"
                        else "apply_patch"
                    ),
                    "runtime": 0,  # to be filled in by the tool wrapper
                }
            ),
        }
        self.turns.append(turn)

    def add_request(self):
        self.requests.append((self.turns, self.request_type))
        self.turns = []

    def dump(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "agent_name": self.agent_name,
                    "requests": self.requests,
                    "cost_analysis": self.compute_cost(),
                    "time_analysis": self.compute_response_times(),
                    "tool_runtime_analysis": self.compute_tool_runtimes(),
                    "tool_count_analysis": self.compute_tool_counts(),
                    "tokens_analysis": self.compute_tokens(),
                },
                f,
                indent=4,
            )

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.agent_name = data["agent_name"]
            self.requests = data["requests"]

    def compute_cost(self):
        types = set([request[1] for request in self.requests])
        cost_per_type = {type: 0 for type in types}
        for request in self.requests:
            turns, request_type = request
            for turn in turns:
                cost_per_type[request_type] += turn["cost"]
        total_cost = sum(cost_per_type.values())
        return cost_per_type, total_cost

    def compute_response_times(self):
        types = set([request[1] for request in self.requests])
        time_per_type = {type: 0 for type in types}
        for request in self.requests:
            turns, request_type = request
            for turn in turns:
                time_per_type[request_type] += turn["response_time"]
        total_time = sum(time_per_type.values())
        return time_per_type, total_time

    def compute_tool_runtimes(self):
        tool_names = set(
            [
                turn["tool"]["name"]
                for request in self.requests
                for turn in request[0]
                if turn["tool"] is not None
            ]
        )
        runtime_per_tool = {tool: 0 for tool in tool_names}
        for request in self.requests:
            turns, request_type = request
            for turn in turns:
                if turn["tool"] is not None:
                    runtime_per_tool[turn["tool"]["name"]] += turn["tool"]["runtime"]
        total_runtime = sum(runtime_per_tool.values())
        return runtime_per_tool, total_runtime

    def compute_tool_counts(self):
        tool_names = set(
            [
                turn["tool"]["name"]
                for request in self.requests
                for turn in request[0]
                if turn["tool"] is not None
            ]
        )
        count_per_tool = {tool: 0 for tool in tool_names}
        for request in self.requests:
            turns, request_type = request
            for turn in turns:
                if turn["tool"] is not None:
                    count_per_tool[turn["tool"]["name"]] += 1
        total_count = sum(count_per_tool.values())
        return count_per_tool, total_count

    def compute_tokens(self):
        types = set([request[1] for request in self.requests])
        tokens_per_type = {
            type: {
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            }
            for type in types
        }
        for request in self.requests:
            turns, request_type = request
            for turn in turns:
                tokens_per_type[request_type]["input_tokens"] += turn["tokens"][
                    "input_tokens"
                ]
                tokens_per_type[request_type]["cached_tokens"] += turn["tokens"][
                    "cached_tokens"
                ]
                tokens_per_type[request_type]["output_tokens"] += turn["tokens"][
                    "output_tokens"
                ]
                tokens_per_type[request_type]["reasoning_tokens"] += turn["tokens"][
                    "reasoning_tokens"
                ]
        total_tokens = {
            "input_tokens": sum(
                [tokens_per_type[type]["input_tokens"] for type in types]
            ),
            "cached_tokens": sum(
                [tokens_per_type[type]["cached_tokens"] for type in types]
            ),
            "output_tokens": sum(
                [tokens_per_type[type]["output_tokens"] for type in types]
            ),
            "reasoning_tokens": sum(
                [tokens_per_type[type]["reasoning_tokens"] for type in types]
            ),
        }
        return tokens_per_type, total_tokens
