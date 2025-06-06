# 📊 Islamic Finance Tools Suite

This repository includes two integrated modules designed to support Islamic finance compliance, education, and automation using AI and financial standards from AAOIFI:

1. **Zakat Calculator** (`zakat_calculator.py`)  
   A comprehensive tool to calculate and evaluate Zakat liabilities based on AAOIFI FAS 9 standards, with AI-generated compliance advice and optimization strategies.

2. **AAOIFI Standards Tutorial** (`tutorial.py`)  
   An interactive, multilingual educational platform that explains AAOIFI accounting standards through real-world examples, AI-generated breakdowns, and feedback on user responses.


## 🔧 Features

### ✅ Zakat Calculator (AAOIFI FAS 9)

- Classifies balance sheet items (zakatable, non-zakatable, deductible, etc.)
- Calculates zakat base and amount due using net asset method
- Validates against Nisab using real-time gold/silver rates
- Generates compliance advice and optimization tips via LLM (ChatGPT)
- Produces PDF documents:
  - Zakat Compliance Certificate
  - Detailed Zakat Report

### 🎓 AAOIFI Standards Tutorial & Explorer

- Learn about key AAOIFI standards: FAS 4, 7, 10, 28, 32
- Interactive tutorial mode: submit answers and get AI-generated feedback
- English and Arabic support
- Custom Q&A tool with AI-powered responses
- Glossary of common Islamic finance terms

---

## 🖥️ Tech Stack

| Layer         | Technology                                 |
|---------------|---------------------------------------------|
| Frontend      | [Streamlit](https://streamlit.io)           |
| Backend       | Python 3.x                                  |
| AI Model      | OpenAI GPT via LangChain                    |
| PDF Reports   | FPDF (for certificate & report generation)  |
| Translation   | `googletrans` API                           |
| Memory        | LangChain ConversationBufferMemory          |
| Data Storage  | In-memory & JSON structures                 |

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.8+
- OpenAI API key (for AI functionality)
- Streamlit

### 📁 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/islamic-finance-suite.git
cd islamic-finance-suite

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

> **Note:** If `requirements.txt` is missing, you can manually install:

```bash
pip install streamlit pandas numpy fpdf langchain openai googletrans python-dotenv
```

### 🔑 Environment Variables

Create a `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## ▶️ Running the Apps

### 1. Run the Zakat Calculator

```bash
streamlit run zakat_calculator.py
```

### 2. Run the AAOIFI Standards Tutorial

```bash
streamlit run tutorial.py
```

---

## 📎 Example Use Cases

* **Finance departments** of Islamic institutions performing automated Zakat audits
* **Students and trainees** learning AAOIFI standards through guided practice
* **Compliance officers** reviewing financial clauses and Shariah evaluations
* **Developers** building AI-driven Shariah compliance tools

---

## 📄 References

* AAOIFI Financial Accounting Standards (FAS)
* Islamic Financial Services Board (IFSB)
* Best practices in Islamic finance auditing and education