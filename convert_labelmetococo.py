import labelme2coco

labelme_folder = "D:/Programacao/Python/PDI_Segmentation/data/Teste/Train"

# set path for coco json to be saved
save_json_path = "D:/Programacao/Python/PDI_Segmentation/data/Teste/Train/train_coco.json"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)

