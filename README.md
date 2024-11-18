# **Chatbot Description**

This repository contains a chatbot designed to answer questions about a firm's details and their medical services. The chatbot uses Flask for the backend and TensorFlow for machine learning functionalities.
  
## **Features**
- Provides detailed information about the firm.
- Describes the medical services offered.

## **Technologies Used**
- **Python**: Programming language.
- **Flask**: Lightweight web framework for backend development.
- **Flask-CORS**: Enables Cross-Origin Resource Sharing (CORS) support.
- **TensorFlow**: Machine learning framework.
- **NLTK**: Natural Language Toolkit for text processing.

## **Setup and Installation**

### **Installation Steps**
1. **Install Python**  
   Make sure Python 3.8 is installed on your system.

2. **Create a Virtual Environment**  
   ```bash
   python -m venv tf_env
   ```

3. **Activate the Virtual Environment**  
   On Windows:  
   ```bash
   tf_env\Scripts\activate
   ```  
   On macOS/Linux:  
   ```bash
   source tf_env/bin/activate
   ```

4. **Install Required Libraries**  
   Install Flask and TensorFlow:
   ```bash
   pip install flask
   pip install tensorflow==2.13.0
   ```

   Install Flask-CORS and NLTK:
   ```bash
   pip install flask-cors
   pip install nltk
   ```

5. **Run the Application**  
   Start the Flask server:
   ```bash
   python app.py
   ```

6. **Access the Chatbot**  
   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## **Usage**
After running the application, use the chatbot interface in your browser to ask about the firm and its medical services. The chatbot is trained to understand and provide meaningful responses about the firm's offerings.
Notice : The chatbot speaks german.

## **Contribution**
If you’d like to contribute:
1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Submit a pull request.
