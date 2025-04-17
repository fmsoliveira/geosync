
"""
Main entry point for the crew.
"""

import os
from pathlib import Path
import sys
import warnings
import datetime

# Add the project root to PYTHONPATH
project_root = Path(__file__).parent.parent.parent
os.environ["PYTHONPATH"] = str(project_root)

from geosync.crew import Geosync

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """Run the crew."""
    # # Get today's date and one week ago
    # today = datetime.datetime.now()
    # one_week_ago = today - datetime.timedelta(days=7)

    # # Format dates as YYYY-MM-DD
    # first_date = one_week_ago.strftime("%Y-%m-%d")
    # second_date = today.strftime("%Y-%m-%d")

    # inputs = {
    #     "address": "Rua Augusta, Lisboa, Portugal",
    #     "first_date": first_date,
    #     "second_date": second_date,
    #     "current_year": str(datetime.datetime.now().year)
    # }
    inputs={
        "address": "Largo dos Colegiais, Évora",
        "first_date": "2024-04-06",
        "second_date": "2024-04-13",
        "current_year": str(datetime.datetime.now().year)
    }
    try:
        Geosync().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Geosync().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Geosync().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test(n_iterations: int, openai_model_name: str):
    """Test the crew."""
    inputs = {
        "address": "Rua Augusta, Lisboa, Portugal",
        "first_date": "2024-04-06",
        "second_date": "2024-04-13",
        "current_year": str(datetime.datetime.now().year)
    }
    try:
        Geosync().crew().test(n_iterations=n_iterations, openai_model_name=openai_model_name, inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    run()
