# Neuromorphic Audio Denoiser App Setup Instructions

Follow these steps to set up and run the project.

## 1. Create a Virtual Environment

Isolate the project dependencies by creating a virtual environment.

```bash
python -m venv venv
```

## 2. Activate the Environment

Activation depends on your operating system:

**On macOS / Linux:**

```bash
source venv/bin/activate
```

**On Windows:**

```bash
.\venv\Scripts\activate
```

## 3. Install Requirements

Install all necessary libraries (Streamlit, Librosa, NumPy, etc.) from the provided list.

```bash
pip install -r requirements.txt
```

## 4. Run the Application

Start the Streamlit server to open the app in your browser.

```bash
streamlit run App.py
```
