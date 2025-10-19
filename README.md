# Interview-Chatbot

Project Overview

The Dentistry Interview Evaluator is an AI-powered web application designed to standardize and accelerate the screening process for dental candidates. It automates the subjective, time-consuming manual evaluation of written interview responses.

The system uses a Generative AI model (via Fireworks API) to analyze responses against expert criteria, providing objective and quantifiable results for every candidate.

Problem Solved

Hiring managers and dentists often spend significant time manually reviewing interview responses, leading to slow hiring cycles and inconsistent scoring. This project aims to automate this crucial phase to:

Standardize Scoring: Ensure every candidate is evaluated consistently using the same criteria.

Accelerate Hiring: Provide rapid, automated scoring and recommendations.

Generate Insights: Deliver structured data, including an Average Score, detailed strengths/weaknesses, and a clear Hire/Reject Recommendation.

Main Features

Secure Admin Workflow: Managers use a secure Admin Panel to generate unique, one-time interview links.

Automated Interview: The candidate receives a unique link and completes a four-question interview designed to test core competencies.

Objective AI Evaluation: The system automatically analyzes all answers against professional criteria.

Data-Driven Results: Results (scores and recommendations) are stored securely in the database and emailed to the administrator.

Report Download: Administrators can download the final, detailed evaluation report from the Admin Panel.

Tools & Technologies (Cloud Ready)

Category

Tools

User Interface

Dash (Plotly) with Bootstrap Components

Logic & Backend

Python, Flask (via Dash server)

AI Integration

Generative AI via Fireworks API

Data Storage

SQLite (Local Development) / PostgreSQL or Firestore (Cloud Deployment)

Containerization

Docker and Docker Compose

Next Steps

Once the application is deployed to a public server, testing and feedback are critical:

Final Testing & Validation

From this Link: http://10.24.105.221:8051/admin?pw=My@dmin-Pass!word (Admin panel)

Run Live Interview Simulation

Gather Administrator Feedback

Collect Candidate Experience Feedback

Future Enhancements

Candidate Time Logging

Live Admin Dashboard

Support for Multiple Admins

Customizable Core Topics
