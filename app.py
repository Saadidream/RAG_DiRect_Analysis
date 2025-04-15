# app.py
import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load data functions
def load_notes(samples_dir):
    notes = []
    for root, dirs, files in os.walk(samples_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    # Check if data is a list and not empty before accessing element 0
                    if isinstance(data, list) and len(data) > 0:
                        note_data = data[0]  # Assuming one note per file
                    # Handle the case where data is not a list or is empty
                    else:
                        # Adjust based on actual structure; assuming data is a dict if not a list
                        note_data = data  # or note_data = data.get('key_containing_note_data') if it's a dictionary
                    inputs = [note_data.get(f'input{i}', '') for i in range(1, 7)]
                    inputs = [inp if inp != 'None' else '' for inp in inputs]  # Handle missing values
                    sections = ['Chief Complaint', 'History of Present Illness', 'Past Medical History',
                                'Family History', 'Physical Exam', 'Pertinent Results']
                    document = '\n'.join([f"{sec}: {inp}" for sec, inp in zip(sections, inputs) if inp])
                    notes.append(document)
    return notes

def load_knowledge_graphs(diagnostic_kg_dir):
    kg_data = {}
    for file in os.listdir(diagnostic_kg_dir):
        if file.endswith('.json'):
            with open(os.path.join(diagnostic_kg_dir, file), 'r') as f:
                kg_data[file] = json.load(f)
    return kg_data

# Retriever class with local model loading
class DenseRetriever:
    def __init__(self, documents, model_name='all-MiniLM-L6-v2'):
        self.documents = documents
        local_model_path = 'models/all-MiniLM-L6-v2'
        try:
            if os.path.exists(local_model_path):
                st.write("Loading SentenceTransformer model from local path...")
                self.model = SentenceTransformer(local_model_path)
            else:
                st.write("Local model not found, attempting to download SentenceTransformer model...")
                self.model = SentenceTransformer(model_name)
        except Exception as e:
            st.error(f"Failed to load SentenceTransformer model: {str(e)}")
            raise e
        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)
    
    def get_top_k(self, query, k=3):
        query_embedding = self.model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.documents[i] for i in indices[0]], distances[0]

# Summarizer
def summarize_documents(documents, max_length=150):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    for i, doc in enumerate(documents):
        doc = doc[:1000]
        summary = summarizer(doc, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(f"Patient Case {i+1}: {summary}")
    return summaries

# Generator class
class Generator:
    def __init__(self, model_name='google/flan-t5-base'):
        local_model_path = 'models/google-flan-t5-base'
        try:
            if os.path.exists(local_model_path):
                st.write("Loading Flan-T5 model from local path...")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
            else:
                st.write("Local Flan-T5 model not found, attempting to download...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            st.error(f"Failed to load Flan-T5 model: {str(e)}")
            raise e
    
    def generate_answer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_length=200, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Knowledge helpers
def extract_symptom(query):
    if "with" in query.lower():
        parts = query.lower().split("with")
        if len(parts) > 1:
            symptom = parts[1].strip()
            if symptom.endswith("?"):
                symptom = symptom[:-1]
            return symptom
    return "unknown symptom"

def find_relevant_diseases(symptom, knowledge_graphs, synonyms):
    relevant_diseases = []
    symptom_lower = symptom.lower()
    for disease_file, kg in knowledge_graphs.items():
        for step, data in kg["knowledge"].items():
            if "Symptoms" in data:
                symptoms = data["Symptoms"].lower()
                if (symptom_lower in symptoms or 
                    any(syn.lower() in symptoms for syn in synonyms)):
                    relevant_diseases.append(disease_file)
                    break
    return relevant_diseases

def format_knowledge(relevant_diseases, knowledge_graphs):
    knowledge_text = ""
    for disease_file in relevant_diseases:
        disease_name = file.replace('.json', '').replace('_', ' ')
        knowledge_text += f"For {disease_name}:\n"
        kg = knowledge_graphs[disease_file]
        for step, data in kg["knowledge"].items():
            if isinstance(data, str):
                knowledge_text += f"- {step}: {data}\n"
            elif isinstance(data, dict):
                for key, value in data.items():
                    knowledge_text += f"- {key}: {value}\n"
        knowledge_text += "\n"
    return knowledge_text.strip()

# RAG Pipeline
class RAGPipeline:
    def __init__(self, retriever, generator, knowledge_graphs=None):
        self.retriever = retriever
        self.generator = generator
        self.knowledge_graphs = knowledge_graphs
        self.synonyms = ["shortness of breath", "dyspnea", "breathlessness", "sob", "difficulty breathing"]
    
    def answer_query(self, query, k=3):
        symptom = extract_symptom(query)
        relevant_diseases = find_relevant_diseases(symptom, self.knowledge_graphs, self.synonyms)
        knowledge_text = format_knowledge(relevant_diseases, self.knowledge_graphs) if relevant_diseases else "No specific diagnostic criteria available."
        
        retrieved_docs, distances = self.retriever.get_top_k(query, k)
        summaries = summarize_documents(retrieved_docs)
        context = '\n\n'.join(summaries)
        
        prompt = (
            f"Based on the following patient cases and diagnostic criteria for diseases associated with '{symptom}', "
            f"list the possible diagnoses for a patient presenting with '{symptom}'. For each diagnosis, briefly explain "
            f"the supporting evidence from the patient cases or diagnostic criteria.\n\n"
            f"Patient Cases:\n{context}\n\n"
            f"Diagnostic Criteria:\n{knowledge_text}\n\n"
            f"Possible Diagnoses (format as a bullet list with evidence):"
        )
        
        answer = self.generator.generate_answer(prompt)
        return retrieved_docs, summaries, answer, distances

# Cache data and resources
@st.cache_data
def load_notes_cached(samples_dir):
    return load_notes(samples_dir)

@st.cache_data
def load_knowledge_graphs_cached(diagnostic_kg_dir):
    return load_knowledge_graphs(diagnostic_kg_dir)

@st.cache_resource
def load_retriever(documents):
    return DenseRetriever(documents)

@st.cache_resource
def load_generator():
    return Generator()

# Initialize
try:
    samples_dir = 'data/samples'
    diagnostic_kg_dir = 'data/diagnostic_kg'
    documents = load_notes_cached(samples_dir)
    knowledge_graphs = load_knowledge_graphs_cached(diagnostic_kg_dir)
    retriever = load_retriever(documents)
    generator = load_generator()
    pipeline = RAGPipeline(retriever, generator, knowledge_graphs)
except Exception as e:
    st.error(f"Failed to initialize the pipeline: {str(e)}")
    st.stop()

# Streamlit UI
st.title("Clinical RAG System for Diagnosis")
st.write("Enter a clinical query to get possible diagnoses based on patient cases and diagnostic criteria.")
query = st.text_input("Query (e.g., 'What is the diagnosis for a patient with shortness of breath?'):")

if st.button("Get Diagnoses"):
    if query:
        try:
            retrieved_docs, summaries, answer, distances = pipeline.answer_query(query)
            st.subheader("Retrieved Patient Cases")
            for i, (doc, dist) in enumerate(zip(retrieved_docs, distances)):
                with st.expander(f"Document {i+1} (Distance: {dist:.2f})"):
                    st.write(doc)
            st.subheader("Summarized Patient Cases")
            for summary in summaries:
                st.write(summary)
            st.subheader("Possible Diagnoses")
            st.write(answer)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    else:
        st.warning("Please enter a query.")
