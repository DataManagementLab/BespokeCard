import logging

from llm_cache import setup_logging
from utils.plots import create_plots_directories
from utils.semantic_analysis import run_semantic_analysis
from utils.subplan_generation import (
    generate_subplans,
    annotate_subplans_with_true_cards,
    annotate_subplans_with_pg_cards,
)
from local_agents import coding_loop

if __name__ == "__main__":
    setup_logging(logging.INFO)
    create_plots_directories()
    with open("card_estimator.py", "w") as f:
        f.write("# to be implemented\n")
    with open("outputs/feedback.json", "w") as f:
        f.write("{}")
    with open("outputs/outliers.json", "w") as f:
        f.write("[]")
    generate_subplans()
    annotate_subplans_with_true_cards()
    annotate_subplans_with_pg_cards()
    coding_loop()
    run_semantic_analysis("outputs")
