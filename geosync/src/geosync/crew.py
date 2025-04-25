from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# import tools
from geosync.tools.geocoding_tool import GeoapifyTool
from geosync.tools.earthengine_tool import EarthEngineImageFetcherTool
from geosync.tools.image_difference_analyzer_tool import ImageDifferenceAnalyzerTool
#from geosync.tools.urban_analysis_tool import UrbanGrowthAnalyzerTool

import os

load_dotenv()

print("CREWAI_TELEMETRY_DISABLED:", os.getenv("CREWAI_TELEMETRY_DISABLED"))


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Geosync():
    """Geosync crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    @agent
    def geocoder_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['geocoder_agent'],
            tools=[GeoapifyTool()],
            llm=ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.5,
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            verbose=True
        )

    @agent
    def satellite_image_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['satellite_image_agent'],
            tools=[EarthEngineImageFetcherTool()],
            llm=ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.4,
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            verbose=True
        )

    @agent
    def image_analysis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['image_analysis_agent'],
            tools=[ImageDifferenceAnalyzerTool()],
            llm=ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            verbose=True
        )

    # @agent
    # def urban_growth_agent(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['urban_growth_agent'],
    #         tools=[UrbanGrowthAnalyzerTool()],
    #         llm=ChatOpenAI(
    #             model="gpt-3.5-turbo",
    #             temperature=0.3,
    #             api_key=os.getenv("OPENAI_API_KEY")
    #         ),
    #         verbose=True
    #     )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task

    @task
    def geocode_task(self) -> Task:
        return Task(
            config=self.tasks_config['geocode_task']#,
            #inputs={"address": "Largo dos Colegiais, Évora"}
        )

    @task
    def fetch_satellite_image_task(self) -> Task:
        return Task(
            config=self.tasks_config['fetch_satellite_image_task']
        )

    @task
    def analyze_image_differences_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_image_differences_task']
        )

    # @task
    # def urban_growth_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['urban_growth_task']
    #     )

    @crew
    def crew(self) -> Crew:
        """Creates the Geosync crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
