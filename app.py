from fastai.vision.all import *
import gradio as gr
import os
from apikey import apikey
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

title = "CropDoc"
description = " Upload a picture of a Soybean plant leaf or drag and drop one of the examples below to the upload box to find out if the soybean plant is suffering from a disease or if it's healthy!\n" 
with gr.Blocks() as demo:
    with gr.Tab("Disease identifier"):
        learn = load_learner('modelbest1.pkl')

        categories = ('Brown Spot','Bacterial Blight', 'Bacterial Pustule','Frogeye Leaf Spot', 'Cercospora Leaf Blight','Downy Mildew', 'Healthy Soybean Leaf')
        def classify_image(img):
            pred,idx,probs = learn.predict(img)
            return dict(zip(categories, map(float,probs)))

        # def classify_image(img):
        #     pred, idx, probs = learn.predict(img)
        #     max_prob_idx = torch.argmax(probs).item()
        #     max_prob_category = categories[max_prob_idx]
        #     max_prob = float(probs[max_prob_idx])
        #     return {max_prob_category: max_prob}


        image = gr.Image(shape=(192,192))
        label = gr.Label()
        examples = ['BacterialBlight.jpg','BrownSpot.jpg','DownyMildew.jpeg','FrogeyeLeafSpot.jpg','HealthyLeaf.jpg']
        #intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, title=title, description=description)

        intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples = examples, title=title, description=description)
        #intf.launch(inline=False)
    with gr.Tab("CropDoc Bot"):


        # Set the apikey
        os.environ['OPENAI_API_KEY'] = 'apikey'

        # Initializing the LLM and the Toolset
        llm = OpenAI(temperature=0)
        tools = load_tools(["wikipedia"], llm=llm)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

        # Create an empty conversation history
        conversation_history = []

        def langchain_bot(prompt):
            if prompt:
                # Add the user's message to the conversation history
                conversation_history.append({"role": "user", "content": prompt})
                # Generate a response based on the entire conversation history
                text = agent.run(conversation_history)
                # Add the bot's response to the conversation history
                conversation_history.append({"role": "bot", "content": text})
                return text
            return "Type a prompt to learn about the plant disease!"




        # Customize the appearance and style of the interface
        iface = gr.Interface(fn=langchain_bot, inputs=gr.inputs.Textbox(lines=5, placeholder="Type your prompt here..."), 
                        outputs=gr.outputs.Textbox(), title="CropDoc Bot",
                        description="Ask CropDoc about the disease affecting your crops")

        # Launch the interface
        # iface.launch()

        demo.launch()

