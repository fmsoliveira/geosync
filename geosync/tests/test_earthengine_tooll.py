
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
    input_json = json.dumps({
        "lat": 38.7169,
        "lon": -9.1399,
        "formatted_address": "Lisboa, Portugal"
    })

    input_two = json.dumps({
        "input_data": "38.7169,-9.1399,2025-04-06,2025-04-13"
    })

    input_three = json.dumps({
        "lat": 37.42315153585935, "lon": -122.08350820058713
    })

    input_four = "The coordinates are: {\"lat\": 38.5732763, \"lon\": -7.9058921, \"formatted_address\": \"...\"}"
    
    try:
        print(f"A executar a tool com input:")    
        logging.info(f"A executar a tool com input:")
        
        resultado = tool._run(38.5732763, -7.9058921)
        
        print(f"--- Resultado da Tool ---")
        print(resultado)
        logging.info(f"Resultado da tool: {resultado}")
        
    except Exception as e:
        print(f"Erro durante a execução da tool: {e}")
        logging.error(f"Erro durante a execução da tool:", exc_info=True) # Loga o traceback
        
    print("--- Fim do teste isolado da Tool ---")