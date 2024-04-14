"""
CS 437 Lab 5: Wildlife Conservation Data Extraction Script
Author: Blake McBride (blakepm2@illinois.edu)
Purpose: To extract data from the READABLE simulation logs into a clean .json format and facilitate analysis
"""

# import re for regex and json for saving the data
import json
import re

# method for parsing the GPS data, temperature, oxygen_saturation, heart_rate, and air_quality at each timestamp for each distinct animal using regex
def parse_animal_data(file_path: str) -> dict:
    """
    Parses GPS data, temperature, oxygen_saturation, heart_rate, and air_quality at each timestamp for each distinct animal from a READABLE simulation log file and returns it as a dictionary
    
    This function reads a file from the World_win/StandalongWindows64/SavannaLogs/Simulation/simulation_2023_xx_xx_xx_xx_READABLE.txt format and extracts the timestamp, GPS coordinates, temperature, oxygen_saturation, heart_rate, and air_quality recorded for each distinct animal; formatting the results into a dictionary which can then be processed into a clean .json file at the end of this script.

    Args:
        file_path (str): The path to the *READABLE* .txt file containing the data for your animals.

    Returns:
        dict: A dictionary containing all parsed data which can be saved as a clean .json and/or loaded into a pandas DataFrame for further analysis.
    """
    # setup attributes for storing data and keeping track of the current (distinct) animal
    data = {}
    current_animal = None

    # define some regular expression patterns for grabbing the animal id, the timestamp, gps coordinates, temperature, oxygen_saturation, heart_rate, and air_quality
    animal_pattern = re.compile(r'^=============== (.*?):(.*?) LOG ===============$')
    timestamp_pattern = re.compile(r'-- Timestamp: (\[\d.]+)')
    data_pattern = re.compile(r'"(\w+)": (\[?-?\d+\.?\d*(?:,\s*-?\d+\.?\d*)?\]?)')

    # open the *READABLE* simulation log file
    with open(file_path, 'r') as file:
        for line in file:
            # check for animal type and id and create new entries if (new) distinct animal
            animal_match = animal_pattern.match(line.strip())
            if animal_match:
                animal_type = animal_match.group(1)
                animal_id = animal_match.group(2)
                current_animal = f"{animal_type}:{animal_id}"
                # set up attributes reflecting the timestamps and other data for the distinct animal represented as lists
                data[current_animal] = {"timestamp": [], "gps coordinates": [], "temperature": [], "oxygen_saturation": [], "heart_rate": [], "air_quality": []}
                continue

            # check for timestamp and other data if current_animal
            if current_animal:
                timestamp_match = timestamp_pattern.match(line.strip())
                data_match = data_pattern.findall(line.strip())

                # if there is a new timestamp, we add it to the list of timestamps for the distinct animal
                if timestamp_match:
                    current_timestamp = timestamp_match.group(1)
                    data[current_animal]["timestamp"].append(current_timestamp)

                # if there is data, we add it to the appropriate list for the distinct animal
                for key, value in data_match:
                    if key == "location":
                        data[current_animal]["gps coordinates"].append(json.loads(value))
                    else:
                        data[current_animal][key].append(float(value))

    return data


# example usage 
#TODO fill in your infile and outfile paths
infile_path = r"C:\Users\Bryan\Downloads\World_win\StandaloneWindows64\SavannaLogs\Simulation\simulation_2024_4_13_18_56_39_READABLE.txt"
outfile_path = r"C:\Users\Bryan\Downloads\World_win\StandaloneWindows64\SavannaLogs\SimulationResult1"

# parse the READABLE simulation log
parsed_data = parse_animal_data(infile_path)

# save the loaded data into a clean .json file
with open(f"{outfile_path}.json", 'w') as outfile:
    json.dump(parsed_data, outfile, indent=4)
outfile.close()