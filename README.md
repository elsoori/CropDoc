# CropDoc
 Your Toolkit to Growing Healthier Fields
 In this project I classified images of 6 of the most common foliage diseases in soybeans as well as healthy soybean leaves(so there are 7 categories), and I have made the CropDpc ChatBot that retrieves information from Wikipedia about all the crop diseases you want to know about and also remembers the context of the message so that you can ask follow up questions.
This project frontend was completed using Gradio and it is hosted on Hugging Spaces.

Please find the app on: https://huggingface.co/spaces/elsoori/CropDocV1
Please note that you have to duplicate the space and use your OpenAI API key, where API key is written on the app.py file of the CropDocV1 hosted on Huggingface spaces.
The project I completed has an Image Classification as well as a Large Language model with Retrieval AUgmented Generation(LLM with RAG).

In the Image Classification portion of the project, I used deep learning with Pytorch and Fastai to train and fine-tune on Pytorch Image Models(timm). I intially started with resnet34 model while I was troubleshooting and then  I used timm models as GPU resources allowed. After first training the model, I cleaned it and fine-tuned it. 
There were no publicly available datasets for this. Therefore, I used duckduckgo image search to scape the images. I used a 80:20, training set to validation set data ratio and fine-tuned for 3-4 epochs. I made sure to not overfit the data. As you would notice in the huggingface spaces, there are many model.pkl files from the many attempts at getting the best model. The biggest challenge was that the labeling on the search engined were inaccurate. The dataset size for each type of disease classified was 100 images.

In the CropDoc ChatBot side of things, a Large Language model with Retrieval Augmented Generation(LLM with RAG) has been used. I used OpenAI and to avoid hallucinations and to receive up-to-date information, I decided to use Wikipedia API. I was intially going to make a knowleddge base by scraping websites on the 6 diseases I have classified, However I thought it would be much better for scalability as well as the upto date information, to just use Wikipedia. That is the reason you will see the the CSV file I created by scraping the webpage text and fitting it into Pandas Dataframe. I used beautifulsoup to scape the text and I was using Chroma for indexing purposes to embed the knowledge base into OpenAI, when I decided to use Wikipedia.
