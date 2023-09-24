# CropDoc
 Your Toolkit to Growing Healthier Fields
 In this project I classified images of 6 of the most common foliage diseases in soybeans, and I have made the CropDpc ChatBot that retrieves information from Wikipedia about all the crop diseases you want to know about and also remembers the context of the message so that you can ask follow up questions.

Please find the app on: https://huggingface.co/spaces/elsoori/CropDocV1
Please note that you have to duplicate the space and use your OpenAI API key, where API key is written on the app.py file of the CropDocV1 hosted on Huggingface spaces.
The project I completed has an Image Classification as well as a Large Language model with Retrieval AUgmented Generation(LLM with RAG).

In the Image Classification portion of the project, I used deep learning with Pytorch and Fastai to train and fine-tune on Pytorch Image Models(timm). I intially started with resnet34 model while I was troubleshooting and then  I used timm models as GPU resources allowed. After first training the model, I cleaned it and fine-tuned it. 
There were no publicly available datasets for this. Therefore, I used duckduckgo image search to scape the images. I used a 80:20, training set to validation set data ratio, and I fine-tuned for 3-4 epochs. I made sure to not overfit the data. As you would notice in the huggingface spaces, there are many model.pkl files from the many attempts at getting the best model. The biggest challenge was that the labeling on the search engined were inaccurate. The dataset size for each type of disease classified was 100 images.
