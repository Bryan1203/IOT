# Import packages
import os
import argparse
import cv2
import threading
import time
import numpy as np
import sys
import importlib.util
from gtts import gTTS
from playsound import playsound

def play_alert(side):
    print(f"Play alert called for {side}")
    text = f'Pothole {side}'
    tts = gTTS(text=text, lang='en')
    temp_file = f'temp_{side.lower()}.mp3'
    tts.save(temp_file)
    try:
        playsound(temp_file)
    except Exception as e:
        print(f"Failed to play sound: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
