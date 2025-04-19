from geosync.tools.urban_analysis_tool import UrbanGrowthAnalyzerTool

if __name__ == "__main__":
    # create a class of urban analyzer
    urban_analyzer = UrbanGrowthAnalyzerTool()
    
    # run the urban analyzer
    result = urban_analyzer._run("output/rgb_antiga.png", "output/rgb_recente.png")
    print(result)