import json
import logging
import sys
import greengrasssdk

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

client = greengrasssdk.client("iot-data")

global_max_CO2 = 0.0

def lambda_handler(event, context):
    global global_max_CO2

   
    try:
      record = json.loads(event)
      CO2_val = float(record['vehicle_CO2'])
      vehicle_id = record['vehicle_id']

      if CO2_val > global_max_CO2:
          global_max_CO2 = CO2_val

      client.publish(
          topic="iot/Vehicle_" + vehicle_id,
          queueFullPolicy="AllOrException",
          payload=json.dumps({"max_CO2": global_max_CO2}),
      )
    
      return {"max_CO2": global_max_CO2}

    except json.JSONDecodeError as e:
        logger.error("Decoding JSON has failed: %s", str(e))
        return {"error": "Decoding JSON has failed"}

    except KeyError as e:
        logger.error("Missing expected key in JSON: %s", str(e))
        return {"error": "Missing expected key in JSON"}

    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))
        return {"error": "An unexpected error occurred"}

