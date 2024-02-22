# Importing necessary libraries
import os
import pandas as pd
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai


# Setting API key
os.environ['OPENAI_API_KEY'] = ''

# Defining functions for one-shot example
def impression_generation_one_shot_prompt1(model, findings):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': "I will give you the findings of CT chest scans. You are a radiologist, and your task is to write an abstractive summary impression for the report."},
            {'role': 'user', 'content': f"Here is the finding {findings}. Please write a short and concise abstractive impression no longer than one sentence."},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

def impression_generation_one_shot_prompt2(model, findings):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': "Assume the role of a seasoned radiologist or healthcare professional providing a comprehensive and insightful interpretation of the imaging results."},
            {'role': 'user', 'content': f"Here is the finding {findings}. Concisely annotate radiological findings to predict the impression and write it in no longer than one sentence."},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

# Defining functions for few-shot example
def impression_generation_few_shot_prompt1(model, findings, ex1_finding, ex1_impression):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': "I will give you the findings of CT chest scans. You are a radiologist, and your task is to write an abstractive summary impression for the report."},
            {'role': 'user', 'content': f"Here is the CT Chest Scan report, please write a short and concise abstractive impression no longer than one sentence {ex1_finding}"},
            {'role': 'user', 'content': f"The impression is the following {ex1_impression}"},
            {'role': 'user', 'content': f"Here is the finding {findings}. Please write a short and concise impression no longer than one sentence."},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

def impression_generation_few_shot_prompt2(model, findings, ex1_finding, ex1_impression):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': "Assume the role of a seasoned radiologist or healthcare professional providing a comprehensive and insightful interpretation of the imaging results."},
            {'role': 'user', 'content': f"Please Concisely annotate radiological findings to predict the impression and write it in no longer than one sentence. Here are the findings from the chest CT report: {ex1_finding}"},
            {'role': 'user', 'content': f"The impression is the following: {ex1_impression}"},
            {'role': 'user', 'content': f"Here are the findings of a report from another patient: {findings}. Please Concisely annotate radiological findings to predict the impression and write it in no longer than one sentence."},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

# Defining function for RAG impression generation
def RAG_impression_generation(data):
    extractions = []
    gpt_results = []
    for k, row in data.iterrows():
        # Initializing Pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        index_name = "impression"
        clinical_text = row['findings']
        index_name = "impression"
        index = pinecone.Index(index_name)
        doc = Document(page_content=clinical_text, metadata={'patient ID': row['MRN']})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        texts = text_splitter.split_documents([doc])
        llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
        vectorstore = Pinecone.from_documents(texts, instructor_embeddings, index_name=index_name)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, model='gpt-3.5-turbo'), retriever, metadata={'patient ID': row['MRN']})
        chat_history = []
        for question in questions:
            retrieved_documents = retriever.get_relevant_documents(question)
            answer = qa({"question": f"{question}", "chat_history": chat_history})["answer"]
            chat_history.append((question, answer))
        results = [chat[1] for chat in chat_history]
        analysis = " ".join(results)
        extractions.append(results)
        messages = [
            {'role': 'system', 'content': "Assume the role of a seasoned radiologist or healthcare professional providing a comprehensive and insightful interpretation of the imaging results."},
            {'role': 'user', 'content': f"Reading the findings of a radiological report, here is what you found: {analysis}. Based on your analysis, please summarize the information and write a concise impression no longer than one sentence"},
        ]
        gpt_impression_summary = summarize(messages)
        gpt_results.append(gpt_impression_summary)
        index.delete(delete_all=True)
    data['RAG_impression'] = gpt_results
    data['extraction'] = extractions
    return data

# Example findings
ex1_finding = """
Almost unchanged large lobulated anterior mediastinal mass invading and totally occluding the SVC, right brachiocephalic vein, the distal left brachiocephalic vein, and possibly the origin of the right subclavian vein. Secondary dilatation of the azygos/hemiazygos veins.

There is, however, interval improvement of secondary thrombosis involving the lower half of the right internal jugular vein; there is now partial nonocclusive thrombosis.

Otherwise no mediastinal lymph node enlargement.

Less prominent mildly enlarged bilateral axillary lymph nodes measuring up to 1.7 cm in long axis compared to 2.3 cm previously.

No pulmonary focal lesions or consolidation.

Mild amount of pericardial effusion.

No pleural effusion.
"""

ex1_impression = """
Stable anterior mediastinal mass with persistent total occlusion of the SVC, right brachiocephalic vein, and the distal left brachiocephalic vein.

Partial improvement of right internal jugular vein thrombosis.

Less prominent bilateral axillary lymph nodes.
"""

# Reading data
data = pd.read_csv('your_data.csv')  # Update with your file name

# Generating impressions using different methods
data['gpt_impression_zero_prompt1'] = data.apply(lambda row: impression_generation_one_shot_prompt1('gpt-4', row['findings']), axis=1)
data['gpt_impression_zero_prompt2'] = data.apply(lambda row: impression_generation_one_shot_prompt2('gpt-4', row['findings']), axis=1)
data['gpt_impression_few_shot_prompt1'] = data.apply(lambda row: impression_generation_few_shot_prompt1('gpt-4', row['findings'], ex1_finding, ex1_impression), axis=1)
data['gpt_impression_few_shot_prompt2'] = data.apply(lambda row: impression_generation_few_shot_prompt2('gpt-4', row['findings'], ex1_finding, ex1_impression), axis=1)
data_with_rag_impressions = RAG_impression_generation(data)

# Saving to CSV
data_with_rag_impressions.to_csv('cleaned_data_with_impressions.csv', index=False)
