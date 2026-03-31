# Project Context: Fraud Detection ML Pipeline

Read this file before doing anything else. 

You may update this file as you go along. 

## Overview
This project focuses on building a machine learning solution to identify fraudulent transactions. We are strictly adhering to the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework as outlined in the provided textbook documentation.

## Tech Stack
* **Modeling:** Python (Jupyter Notebook)
* **Framework:** CRISP-DM
* **Model Format:** Serialized Pickle file (`model.sav`)
* **Database/Backend:** Supabase
* **Hosting/Deployment:** Vercel

---

## Project Phases (CRISP-DM)
We will document and execute the following phases within the `fraud_detection.ipynb` file:

1.  **Business Understanding:** Define fraud detection goals and success metrics.
2.  **Data Understanding:** Explore the provided dataset and identify quality issues.
3.  **Data Preparation:** Clean, transform, and handle class imbalance (Fraud vs. Non-Fraud).
4.  **Modeling:** Select and train algorithms to predict the target variable.
5.  **Evaluation:** Validate the model against business objectives (Focusing on Recall and F1-Score).
6.  **Deployment:** Exporting `model.sav` for integration with the Supabase/Vercel stack.

---

## File Structure
* `data/` - Directory containing the raw fraud dataset.
* `CRISP-DM_Textbook.md` - The methodological guide for this project.
* `fraud_detection.ipynb` - The primary notebook for development.
* `model.sav` - The final trained model ready for deployment.

---

## Deployment Workflow
1.  **Serialization:** The final model is saved as `model.sav`.
2.  **Backend Integration:** Connect to **Supabase** for data persistence and user management.
3.  **Production:** Deploy a serverless function or API on **Vercel** to serve real-time fraud predictions.