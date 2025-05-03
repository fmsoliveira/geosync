
import json
import os
from crewai import project
import ee

import logging
from geosync.tools.earthengine_tool import EarthEngineImageFetcherTool

# Configuração básica de logging (APENAS para este teste isolado)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    print("--- Iniciando teste isolado da Tool ---")
    
    # Initialize the tool
    tool = EarthEngineImageFetcherTool()

    try:
        ee.Initialize(project="ee-syncearth")
        print("Autenticação bem-sucedida com a conta de serviço!")
    except Exception as e:
        print(f"Erro ao inicializar o Earth Engine: {e}")

    # Test with JSON input
    input_json = {
        "lat": 40.4952269,
        "lon": -3.5733733,
        "first_date": "2023-04-06",
        "second_date": "2024-04-13"
    }

    try:
        print(f"A executar a tool com input:")    
        logging.info(f"A executar a tool com input:")
        
        resultado = tool._run(**input_json)
        
        print(f"--- Resultado da Tool ---")
        print(resultado)
        logging.info(f"Resultado da tool: {resultado}")
        
    except Exception as e:
        print(f"Erro durante a execução da tool: {e}")
        logging.error(f"Erro durante a execução da tool:", exc_info=True) # Loga o traceback
        
    print("--- Fim do teste isolado da Tool ---")