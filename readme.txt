1. run "pip install -r requirements.txt" to install all the requirements
2. download cityscape dataset https://www.kaggle.com/datasets/shuvoalok/cityscapes/data
3. run "python prepare_mylabels.py" to set the label image (change the path for both train and val)
4. run "python inference.py --image_path[image_path] --model_weights_path[model_weight]" to get inference result from single image
5. run "python fintune_model.py" to finetune the model


