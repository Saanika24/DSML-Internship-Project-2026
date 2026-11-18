# AtmoGraph: Supply Chain Ripple Effect Predictor

## About the Project

AtmoGraph is a Data Science and Machine Learning project that focuses on understanding how problems in one part of a supply chain can affect the other connected parts.

In today's business environment, companies depend on multiple suppliers, factories, warehouses, and distributors to deliver their products. Even a small issue, such as a delayed shipment or a supplier shutdown, can create problems across the entire supply chain.

For example, if a supplier fails to deliver raw materials on time, the manufacturing process may slow down. This can further affect warehouse inventory, product distribution, and customer deliveries.

This chain of problems is known as the Supply Chain Ripple Effect.

The main idea behind AtmoGraph is to analyse these connections, understand possible risks, and estimate how a disruption could affect the overall supply chain.

The project uses Python, Data Analysis, Machine Learning, and Graph Theory to develop a system that can help businesses identify potential risks and make better decisions.

## Why I Chose This Project

I wanted to work on a project that connects Data Science and Machine Learning with a real-world business problem.

Supply chain management is an important part of almost every industry. Businesses need to make sure that materials, products, and resources reach the right places at the right time.

However, unexpected situations such as transportation delays, natural disasters, supplier failures, and material shortages can affect business operations.

Through AtmoGraph, I aim to understand these challenges and explore how data analysis and predictive models can help identify and manage supply chain disruptions.

---

## Project Objectives

The main objectives of this project are:

- Understand how a supply chain operates and how different entities depend on each other.
- Collect and analyse relevant supply chain data.
- Clean the dataset and prepare it for further analysis.
- Identify patterns related to supplier performance, delivery delays, and inventory.
- Build a network connecting suppliers, factories, warehouses, and retailers.
- Understand how a disruption can spread through connected entities.
- Develop machine learning models to predict potential supply chain risks.
- Create an interactive dashboard to display analysis and prediction results.

---

## Main Features

### 1. Supply Chain Data Analysis

The first step is to understand the available data.

I will work on collecting, cleaning, and analysing supply chain information to identify useful patterns and relationships.

This includes:
- Handling missing values.
- Removing duplicate records.
- Understanding supplier performance.
- Analysing delivery delays.
- Studying inventory levels.
- Identifying important features for machine learning.

### 2. Supply Chain Network Visualization

A supply chain consists of several connected entities.

In this project, I use graph theory to represent these relationships.

Suppliers, factories, warehouses, distributors, and retailers are represented as nodes, while their connections are represented as edges.

This helps in understanding which entities depend on each other and identifying the possible impact of a disruption.

### 3. Disruption Simulation

One of the main features of AtmoGraph is disruption simulation.

For example, if a supplier suddenly stops operating, the system can analyse the connected supply chain and identify which factories, warehouses, or retailers might be affected.

This helps in understanding how a problem at one location can create further problems across the network.

### 4. Supplier Risk Analysis

Supplier performance plays an important role in supply chain management.

The project aims to analyse factors such as:
- Previous delivery delays.
- Supplier reliability.
- Delivery lead time.
- Order history.
- Availability of alternative suppliers.

These factors can help identify suppliers that may need closer monitoring.

### 5. Machine Learning Predictions

Machine learning will be used to analyse historical supply chain data and make predictions.

The planned prediction tasks include:

- Supplier disruption risk classification.
- Shipment delivery delay prediction.
- Supply chain impact severity prediction.

The models will be trained and evaluated using suitable datasets.

### 6. Interactive Dashboard

A Streamlit dashboard will be developed to make the results easier to understand.

The dashboard will display:
- Supply chain statistics.
- Supplier and factory relationships.
- Disruption simulation results.
- Affected entities.
- Risk levels.
- Prediction results.
- Suggested actions.

---

## Technologies Used

| Technology | How It Is Used |
|---|---|
| Python | Main programming language |
| Pandas | Working with and analysing datasets |
| NumPy | Numerical calculations |
| Matplotlib | Creating charts and graphs |
| Seaborn | Visualizing data patterns |
| Scikit-learn | Building and evaluating ML models |
| XGBoost | Advanced machine learning |
| NetworkX | Creating and analysing supply chain networks |
| Streamlit | Building the interactive dashboard |
| Plotly | Creating interactive visualizations |
| Joblib | Saving and loading trained models |
| Git and GitHub | Managing project code and updates |

---

## Project Folder Structure

```text
AtmoGraph/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── models/
│
├── src/
│
├── reports/
│
├── main.py
├── README.md
└── requirements.txt
```

---

## How to Set Up the Project

Follow these steps to run AtmoGraph on your computer.

### Step 1: Open the Project Folder

Open the AtmoGraph folder in Visual Studio Code.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install the Required Libraries

```bash
pip install -r requirements.txt
```

### Step 5: Start the Application

```bash
streamlit run main.py
```

After running the command, Streamlit will open the application in your browser.

---

## Project Development Progress

I am developing this project in different stages so that each part can be properly implemented and tested.

### Stage 1: Initial Setup
- [x] Created the project structure.
- [x] Developed the initial dashboard.
- [x] Added sample supply chain data.
- [x] Created the basic supply chain network.
- [x] Implemented an initial disruption simulation.

### Stage 2: Data Collection and Analysis
- [ ] Find a suitable supply chain dataset.
- [ ] Understand the dataset and its columns.
- [ ] Clean and preprocess the data.
- [ ] Perform exploratory data analysis.
- [ ] Identify useful features.

### Stage 3: Graph Analysis
- [ ] Create a graph using actual supply chain relationships.
- [ ] Identify important suppliers and connected entities.
- [ ] Analyse network dependencies.
- [ ] Improve disruption propagation analysis.

### Stage 4: Machine Learning
- [ ] Develop a supplier risk prediction model.
- [ ] Develop a delivery delay prediction model.
- [ ] Develop an impact severity prediction model.
- [ ] Evaluate and compare model performance.
- [ ] Save the trained models.

### Stage 5: Dashboard Development
- [ ] Connect the actual dataset.
- [ ] Integrate the machine learning models.
- [ ] Improve network visualizations.
- [ ] Display prediction results.
- [ ] Add useful reports and insights.

### Stage 6: Final Submission
- [ ] Test the complete application.
- [ ] Prepare project documentation.
- [ ] Complete the project report.
- [ ] Prepare the presentation.
- [ ] Demonstrate the final application.

---

## Expected Results

By the end of this project, I aim to develop an application that can:

- Show how different entities are connected in a supply chain.
- Help identify suppliers that may create operational risks.
- Analyse the possible impact of supplier disruptions.
- Estimate shipment delays using machine learning.
- Identify factories, warehouses, and retailers that may be affected.
- Present useful information through an interactive dashboard.

The overall aim is to make supply chain risks easier to understand and help businesses prepare for unexpected disruptions.

---

## Current Project Limitations

The initial version uses sample data to demonstrate how the supply chain network and disruption simulation work.

Real-world dataset integration and machine learning model development are planned as the next stages.

The final prediction results will depend on the quality of the selected dataset and the performance of the trained models.

Any results generated using simulated data will be clearly identified as simulated rather than real-world predictions.

---

## Internship Details

**Organization:** Infotact Solutions

**Internship Role:** Data Associate L1 Intern

**Domain:** Data Science and Machine Learning

**Project Title:** AtmoGraph: Supply Chain Ripple Effect Predictor

**Project Duration:** One Month (project completion duration mentioned in the assignment letter)

---

## Conclusion

AtmoGraph is an opportunity for me to apply the concepts I have learned in Python, Data Analysis, Machine Learning, and Graph Theory to a practical business problem.

While developing this project, I aim to improve my understanding of real-world datasets, machine learning workflows, predictive analysis, and application development.

The project will also help me gain practical experience in building a complete Data Science solution, from understanding the data to presenting the final results.

**Developed as part of the Data Science and Machine Learning internship at Infotact Solutions.**
