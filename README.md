# ATS Resume Scorer

An AI-powered application that helps job seekers optimize their resumes by analyzing them against job descriptions. The tool uses advanced NLP techniques to provide detailed feedback and matching scores.

Live Demo : https://resume-ats-improve.streamlit.app/

# How it works




## 🌟 Features

- Technical skill matching with categorization
- Semantic similarity analysis using BERT
- TF-IDF based content matching
- Detailed feedback and improvement suggestions
- Missing skills identification
- User-friendly Streamlit interface

## 🛠️ Tech Stack

- Python 3.9+
- Streamlit
- spaCy
- Sentence Transformers (BERT)
- scikit-learn
- NumPy

## 📋 Prerequisites

Before running the application, make sure you have Python 3.9 or higher installed on your system. You can check your Python version by running:

```bash
python --version
```

## 🚀 Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ats-resume-scorer.git
cd ats-resume-scorer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 💻 Running the Application

1. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## ☁️ Deploying to Streamlit Cloud

1. Create a Streamlit Cloud account at https://streamlit.io/cloud

2. Connect your GitHub repository to Streamlit Cloud

3. Deploy your app by selecting your repository and branch

4. The requirements.txt file will automatically handle the dependencies

## 📁 Project Structure

```
ats-resume-scorer/
├── README.md
├── requirements.txt
├── streamlit_app.py
├── ats_score.py
└── .gitignore
```

## 🔧 Configuration

The application uses several pre-trained models and configurations:

- spaCy's `en_core_web_sm` for NLP processing
- MPNet (`all-mpnet-base-v2`) for semantic understanding
- Custom technical skill categories defined in `ats_score.py`

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 📝 Usage Tips

1. Paste the complete job description including all requirements
2. Include your entire resume text, keeping formatting minimal
3. Wait for the analysis to complete (may take a few seconds)
4. Review the scores and feedback
5. Make recommended improvements to your resume
6. Re-run the analysis to check your improvements

## ⚠️ Limitations

- The tool works best with plain text input
- Some technical terms or newer technologies might not be in the default categories
- Analysis time may vary based on text length and complexity

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions and feedback, please open an issue in the GitHub repository.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- NLP processing powered by [spaCy](https://spacy.io/)
- Semantic analysis using [Sentence Transformers](https://www.sbert.net/)
