**AI & ML Portfolio Tasks**

This repository contains **5 AI/ML projects** showcasing practical applications of Large Language Models (LLMs), embeddings, vector stores, and AI pipelines. Each task demonstrates different skills, from document retrieval and chatbots to text classification and auto-tagging.

**Table of Contents**

1. [Task 1: Data Analysis / Predictive Modeling](#task-1-data-analysis--predictive-modeling)
2. [Task 2: Image Classification / Computer Vision](#task-2-image-classification--computer-vision)
3. [Task 3: High Availability Web Application](#task-3-high-availability-web-application)
4. [Task 4: Context-Aware Chatbot Using RAG](#task-4-context-aware-chatbot-using-rag)
5. [Task 5: Auto Tagging Support Tickets Using LLM](#task-5-auto-tagging-support-tickets-using-llm)


**Task 1: Data Analysis / Predictive Modeling**

**Objective:**
Perform data analysis and build predictive models on a given dataset to extract insights and predictions.

**Methodology / Approach:**

* Loaded dataset and performed preprocessing (missing value handling, feature scaling, encoding).
* Explored data using visualizations (histograms, scatter plots).
* Built machine learning models such as Linear Regression, Decision Trees, or Random Forest.
* Evaluated models using metrics like accuracy, precision, recall, and RMSE.

**Key Results / Observations:**

* Model achieved X% accuracy (replace X with your result).
* Important features were identified using feature importance plots.
* Insights drawn can help in making informed data-driven decisions.

**Task 2: Image Classification / Computer Vision**

**Objective:**
Build a model to classify images into different categories.

**Methodology / Approach:**

* Used a labeled image dataset and applied preprocessing (resizing, normalization).
* Built a Convolutional Neural Network (CNN) model using PyTorch / TensorFlow.
* Trained the model and applied data augmentation to improve performance.
* Evaluated model using metrics like accuracy, confusion matrix, and F1-score.

**Key Results / Observations:**

* Achieved X% accuracy on the test set.
* Data augmentation improved generalization and reduced overfitting.
* Visualization of intermediate layers helped understand learned features.

**Task 3: High Availability Web Application**

**Objective:**
Design and implement a web application with high availability using load balancing.

**Methodology / Approach:**

* Developed a backend using Python Flask / FastAPI.
* Integrated Nginx as a load balancer to handle multiple requests.
* Implemented horizontal scaling and redundancy for high availability.
* Tested the application with multiple concurrent requests to validate uptime and response times.

**Key Results / Observations:**

* Application can handle concurrent users efficiently.
* Load balancing improved response times and reduced downtime.
* Demonstrates the ability to design robust web architectures.

**Task 4: Context-Aware Chatbot Using RAG**

**Objective:**
Build a conversational AI chatbot that can retrieve information from external documents and maintain context.

**Methodology / Approach:**

* Prepared a custom knowledge base (Wikipedia pages or internal documents).
* Split documents into chunks and created embeddings using HuggingFace or Sentence Transformers.
* Built a Retrieval-Augmented Generation (RAG) pipeline to fetch relevant context.
* Implemented context memory to store conversation history.
* Deployed chatbot using Streamlit for interactive use.

**Key Results / Observations:**

* Chatbot answers questions using relevant document context.
* Maintains conversation context for more natural interactions.
* Useful for FAQ systems or internal knowledge retrieval.

**Task 5: Auto Tagging Support Tickets Using LLM**

**Objective:**
Automatically classify support tickets into categories using a Large Language Model.

**Methodology / Approach:**

* Loaded free-text support ticket dataset and preprocessed text.
* Applied zero-shot learning with LLMs to predict tags.
* Used few-shot learning to improve prediction accuracy by providing example tickets.
* Extracted top 3 predicted tags for each ticket.
* Evaluated results using classification accuracy and top-3 match metrics.

**Key Results / Observations:**

* Zero-shot learning successfully identified main ticket categories.
* Few-shot learning slightly improved top-3 accuracy.
* Approach can scale to large ticket datasets or fine-tuned LLMs.

**Repository Structure**
AI_Portfolio_Tasks/
│
├── Task1_Data_Analysis/
│   └── Task1_Data_Analysis.ipynb
│
├── Task2_Image_Classifier/
│   └── Task2_Image_Classifier.ipynb
│
├── Task3_High_Availability_WebApp/
│   └── Task3_HA_WebApp.ipynb
│
├── Task4_Context_Chatbot/
│   └── Task4_Context_Chatbot.ipynb
│
├── Task5_SupportTicket_Tagging/
│   └── Task5_SupportTicket_Tagging.ipynb
│
└── README.md
**Skills Demonstrated**

* Data preprocessing, visualization, and predictive modeling
* Computer vision & CNN model development
* High availability web applications and load balancing
* Conversational AI, Retrieval-Augmented Generation (RAG)
* Large Language Models, zero-shot & few-shot learning, multi-class classification
hat?
