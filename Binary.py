# training data 
train_master = ["Artificial Intelligence is transforming industries with automation and data processing."]
train_students = [
    "AI is changing businesses by automating tasks and analyzing data.",
    "Automation using AI is impacting many industries positively.",
    "Artificial Intelligence is used in healthcare and finance."
]

# AI-generated content 
ai_generated_samples = [
    "The field of AI is evolving, impacting industries through automation and big data analysis.",
    "Machine intelligence is revolutionizing business operations via automation."
]

# Sample Test Data
test_student_submissions = [
    "AI helps industries automate tasks and analyze large datasets.",
    "Businesses are changing due to artificial intelligence."
]
test_master_copy = ["AI is improving industries by automating tasks and processing large amounts of data."]

import numpy as np
import os
os.system("pip install nltk")
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher 
from textblob import TextBlob
import pickle

nltk.download('punkt')
nltk.download('punkt_tab')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')# eta huggingface transformer


def plagiarism_check(student_text, peer_texts, ai_texts):
    all_texts = peer_texts + ai_texts + [student_text]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity_scores = cosine_similarity(tfidf_matrix)[-1][:-1]  # Last entry vs all others
    max_similarity = max(similarity_scores) if similarity_scores.size > 0 else 0
    return max_similarity  # Closer to 1 means high plagiarism

def relevance_score(student_text, master_text):
    student_embedding = model.encode(student_text)
    master_embedding = model.encode(master_text)
    return cosine_similarity([student_embedding], [master_embedding])[0][0]  # Closer to 1 means high relevance

def section_wise_score(student_sections, master_sections):
    scores = [SequenceMatcher(None, s_text, m_text).ratio() for s_text, m_text in zip(student_sections, master_sections)]
    return scores  # List of section-wise similarity scores


def grammar_check(text):
    corrected_text = str(TextBlob(text).correct())  # Corrected sentence
    mistakes = sum(1 for a, b in zip(text.split(), corrected_text.split()) if a != b)  # Count differences
    return mistakes, corrected_text

print("\n Model Training...")
for student_text in train_students:
    print(f"\n Student Submission: {student_text}")
    print(f" Plagiarism Score: {plagiarism_check(student_text, train_students, ai_generated_samples):.2f}")
    print(f" Relevance Score: {relevance_score(student_text, train_master[0]):.2f}")
    print(f" Specific Section wise Evaluation: {section_wise_score(student_text, train_master)}") # error dicche
    print(f" Grammar checking: {grammar_check(student_text)}")

print("\n Model Testing on New Submissions...")
for student_text in test_student_submissions:
    print(f"\n Student Submission: {student_text}")
    
    # Plagiarism Check
    plagiarism_score = plagiarism_check(student_text, train_students, ai_generated_samples)
    print(f" Plagiarism Score: {plagiarism_score:.2f} ")
    if plagiarism_score < 0.5:
        print(" have not copied that much")
    else:
        print(" copied a lot")

    # Relevance Check
    relevance = relevance_score(student_text, test_master_copy[0])
    print(f" Relevance Score: {relevance:.2f} ")
    if relevance < 0.5:
        print(" below average assignment")
    else:
        print(" good assignment")

    # Section-Wise Score
    student_sections = nltk.sent_tokenize(student_text)
    master_sections = nltk.sent_tokenize(test_master_copy[0])
    section_scores = section_wise_score(student_sections, master_sections)
    print(f" Section-Wise Scores: {section_scores}")

    # Grammar & Spelling
    mistakes, corrected_text = grammar_check(student_text)
    print(f" Grammar Mistakes: {mistakes}, Corrected Text: {corrected_text}")
    if mistakes == 0:
        print(" impressive")
    elif mistakes <=2:
        print(" can do better")
    else:
        print(" needs good amount of improvement")


import streamlit as st

st.title("AI-Powered Student Answer Evaluation")

# Text input areas
st.subheader("Enter Student Submission")
student_text = st.text_area("Paste the student's response here:", "")

st.subheader("Enter Reference Texts")
peer_texts = st.text_area("Paste peer submissions (comma-separated):", "").split(",")
ai_texts = st.text_area("Paste AI-generated content (comma-separated):", "").split(",")

st.subheader("Enter Master Copy (Ideal Answer)")
master_text = st.text_area("Paste the ideal answer (Master Copy):", "")


# **Run Evaluation**
if st.button("Evaluate Submission"):
    if student_text:
        plagiarism = plagiarism_check(student_text, peer_texts, ai_texts)
        relevance = relevance_score(student_text, master_text)
        section_score = section_wise_score(student_text, master_text)
        grammar = grammar_check(student_text)

        st.write(f"**Plagiarism Score:** {plagiarism:.2f} (Closer to 1 = More plagiarized)")
        st.write(f"**Relevance Score:** {relevance:.2f} (Closer to 1 = More relevant to Master Copy)")
        st.write(f"**Section-Wise Score:** {section_score:} (Average relevance per section)")
        st.write(f"**Grammar & Spelling Score:** {grammar:} (Closer to 1 = Fewer errors)")

    else:
        st.warning("Please enter the student's submission.")
