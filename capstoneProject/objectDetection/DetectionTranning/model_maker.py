import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite0')

# train_data_path = 'train/_annotations.csv'
# valid_data_path = 'valid/_annotations.csv'
# test_data_path = 'test/_annotations.csv'

# train_data = object_detector.DataLoader.from_csv(train_data_path)
# validation_data = object_detector.DataLoader.from_csv(valid_data_path)
# test_data = object_detector.DataLoader.from_csv(test_data_path)

train_data, validation_data, test_data = object_detector.DataLoader.from_csv('train/_annotations.csv')

model = object_detector.create(train_data, model_spec=spec, batch_size=1, train_whole_model=True, validation_data=validation_data)

model.evaluate(test_data)

model.export(export_dir='.', tflite_filename='model.tflite')

config = QuantizationConfig.for_float16()
model.export(export_dir='.', tflite_filename='model_quantized.tflite', quantization_config=config)

model.evaluate_tflite('model_quantized.tflite', test_data)