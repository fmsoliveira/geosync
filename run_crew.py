#!/opt/homebrew/anaconda3/envs/geosync/bin/python

from geosync.crew import Geosync

if __name__ == "__main__":
    Geosync().crew().kickoff()
