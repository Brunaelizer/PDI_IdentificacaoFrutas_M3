import cv2
from detectron2.data.datasets import register_coco_instances
import os
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
import matplotlib.pyplot as plt
import matplotlib

register_coco_instances("frutas_train", {}, "data/Teste/Train/train_coco.json", "")
register_coco_instances("frutas_test", {}, "data/Teste/Test/test_coco.json", "")

import random
from detectron2.data import DatasetCatalog, MetadataCatalog
if __name__ == '__main__':
    #Definição das variáveis
    #  Criação da variável de referencia de configuração
    cfg = get_cfg()
    #  Realizando a definição dos datasets e imagem de treinamento
    dataset_dicts = DatasetCatalog.get("frutas_train")
    frutas_metadata = MetadataCatalog.get("frutas_train")

    #Entrada do usuário
    loop = True
    #  Criando loop para verificar se o usuário deseja ou não realizar o treinamento da aplicação
    while loop:
        print("----------------------------------------")
        print("Inicializando a aplicação")
        print("----------------------------------------\n")
        ans = str(input("Deseja realizar o treinamento (S) ou ignorar (N)?\n"))

        if ans == "S" or ans == "s" or ans == "N" or ans == "n":
            loop = False
        else:
            print("\n Entrada Inválida")

    # Retorna o caminho do arquivo de configurações configurado
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # Realiza o trainamento do dataset
    cfg.DATASETS.TRAIN = ("frutas_train",)
    # Seta como "vazio" o dataset de teste no momento do treinamento (teste realizado apenas depois)
    cfg.DATASETS.TEST = ()
    # Configura a velocidade de "trabalho" (muito alta pode causar problemas de memória)
    cfg.DATALOADER.NUM_WORKERS = 4
    # Métodos para criação das informações da inteligência
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    # Salva o progresso no caminho de output configurado
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    #   Caso o usuário deseje realizar o treinamento, chama o método de "training" no training.py
    if(ans == "S" or ans == "s"):
        print("----------------------------------------")
        print("Iniciando treinamento")
        print("----------------------------------------")
        loop = False
        # Seta o "treinador" padrão
        trainer = DefaultTrainer(cfg)
        # Configura o treinanador para iniciar do 0
        trainer.resume_or_load(resume=False)
        # Chama a função de treinamento
        trainer.train()

    # Teste
    #  Cria as configurações de saída
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = ("frutas_test", )
    predictor = DefaultPredictor(cfg)

    # Mostra a imagem gerada
    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = DatasetCatalog.get("frutas_test")
    for d in random.sample(dataset_dicts, 1):
        im = cv2.imread("data/Teste/Test/banana-nanica.png")
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=frutas_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        matplotlib.use('tkagg')
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()

