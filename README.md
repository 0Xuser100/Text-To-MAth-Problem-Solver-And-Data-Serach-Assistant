

---

# 🧮 Math Solver & Knowledge Assistant

Welcome to **Math Solver & Knowledge Assistant**, an interactive Streamlit-based web app that combines math problem-solving capabilities with knowledge-based assistance using advanced language models from Groq. 

This project allows users to input queries, get detailed solutions to mathematical problems, or retrieve relevant information from Wikipedia. It also features a chat history download option for saving conversations.

---

## 🚀 Features

- **Math Solver**: Solve mathematical problems with detailed, step-by-step explanations.  
- **Knowledge Assistant**: Search for information on various topics using the integrated Wikipedia tool.  
- **Multi-Model Support**: Select from advanced language models such as:
  - `gemma2-9b-it`
  - `gemma-7b-it`
  - `llama3-groq-70b-8192-tool-use-preview`
- **Interactive Chat**: A dynamic chat interface where users can interact with the assistant.  
- **Chat History Download**: Save all your interactions as a text file for future reference.  
- **Streamlit-Based UI**: User-friendly and visually appealing interface.

---

## 📦 Installation

1. **Clone the Repository**:
   ```bash
   git clone [math-solver-knowledge-assistant](https://github.com/0Xuser100/Text-To-MAth-Problem-Solver-And-Data-Serach-Assistant.git)
   cd Text-To-MAth-Problem-Solver-And-Data-Serach-Assistant
   ```

2. **Install Dependencies**:
   Make sure you have Python 3.9+ installed. Run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Groq API Key**:
   Obtain your API key from [Groq](https://groq.com/) and have it ready for use in the app.

---

## ▶️ Usage

1. **Run the App**:
   ```bash
   streamlit run app.py
   ```
2. **Open in Browser**:
   The app will automatically open in your default browser. If not, navigate to `http://localhost:8501`.

3. **Interact**:
   - Input your **Groq API Key** in the sidebar.
   - Enter a math or knowledge-based query in the chat interface.
   - Select your preferred model from the dropdown.
   - View detailed responses in real-time and download your chat history if needed.

---

## 🖼️ App Preview

![App Screenshot](https://files.fm/f/yr55nj8769)  

---

## 📋 Example Queries

- **Math Problems**: 
  - "Solve for x in 3x + 5 = 11."
  - "What is the square root of 256?"
- **Knowledge Questions**:
  - "Who was Albert Einstein?"
  - "Explain the process of photosynthesis."

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features or bug fixes, please follow these steps:

1. Fork the repository.
2. Create a new branch (`feature-new-feature`).
3. Commit your changes and push them.
4. Submit a Pull Request.

---

## 🛠️ Technologies Used

- **Python**: Core programming language.  
- **Streamlit**: Interactive UI for the app.  
- **LangChain**: Framework for building LLM applications.  
- **Groq**: Advanced language models for processing user queries.  
- **WikipediaAPIWrapper**: For searching Wikipedia topics.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- **Groq** for providing powerful language models.
- **Streamlit** for its intuitive and efficient web app framework.
- **LangChain** for simplifying LLM-based application development.

---

