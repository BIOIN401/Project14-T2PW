Install all necessaary dependencies using the requirements.txt file
Use a virtual environment to install dependencies to not cause conflitcs with other projects

1. Instructions for creating a venv (virtual environment) 
In the project root run 

python -m venv .venv

Then activate
.\.venv\Scripts\activate.bat

To deactivate 
deactivate

Once inside venv ( shows like this (.venv) C:\Users) 
downlaod all dependencies by the command 
pip install -r requirements.txt


2. Downlaoding the LLM 
We wil be using the Llama 3.3 70B model as our main model however due to speed constraints
We wil use the Llama 3.1 8B for the testing of the pipeline.
You will need a half decent GPU to run the model, or like cloud computing credits 

Download LM studio 
Download llama 3.1 8B and 3.3 70B  (leave this for now if you cant run it might ned to borrow gpu)
for now just get 3.1 8B 
Go to developer section and start the server, and update the Model and URL in the env file 

Now fire up the venv and install all dependencies as above.
Into llm_client.py hard code the requried variables from the LM studio server 

3. Running the LLM 
In Command prompt run the streamlit app from the venv
    streamlit run app.py 
This will open a browser tab that you can enter text in and get a nicer format


4. Next steps 
    Need to build out the secondary prompt that will run when we need to fill gaps
    The method would be like we run the intial extractor pwml_system.txt 
    And then from there we can run the QA graph to find orphans 
    Then input that into our secondary run of another LLM instance perhaps we can keep memory, depending on context windows 
    Then we can check for OA again and have a retry pipeline to fill in gaps 
    Then we will have to label where our connections are from ofc. 