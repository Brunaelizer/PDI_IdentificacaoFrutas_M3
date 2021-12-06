import os
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer




