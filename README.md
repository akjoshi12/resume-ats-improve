# ATS Resume Analyzer

## Overview
The ATS Resume Analyzer is a Streamlit-based application that evaluates resumes against job descriptions to determine their Applicant Tracking System (ATS) compatibility. The app analyzes strengths, weaknesses, and irrelevant sections while providing an improved resume suggestion.

## Features
- **Preprocessing**: Cleans resume and job description text.
- **ATS Scoring**: Computes similarity between resume and job description.
- **Strength & Weakness Analysis**: Identifies key areas for improvement.
- **Irrelevant Section Detection**: Flags sections that might reduce relevance.
- **Resume Enhancement**: Suggests an improved version using `better_resume.py`.

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Enter your resume text and job description.
3. Click **Analyze ATS Score** to compute the score and get insights.
4. Click **Generate Improved Resume Suggestion** to get an optimized resume.

## File Structure
- `streamlit_app.py` - Main application file.
- `better_resume.py` - Module for generating improved resumes.
- `utils/` - Contains preprocessing and ATS scoring utilities.
- `requirements.txt` - Lists dependencies required to run the application.

## Requirements
- Python 3.8+
- Streamlit
- Other dependencies (listed in `requirements.txt`)

## Contributing
Contributions are welcome! Feel free to fork the repository, create a branch, and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, please reach out via [Your Contact Info].

