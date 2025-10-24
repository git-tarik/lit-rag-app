# LIT-RAG: Lightning-Fast Document Q&A

LIT-RAG is a Streamlit-based application designed to provide lightning-fast answers to questions about your documents. Whether it's a PDF, DOCX, or plain text file, LIT-RAG allows you to upload files or provide URLs, and it retrieves concise answers using a backend Retrieval-Augmented Generation (RAG) system.

---

## Features
- **File Uploads**: Upload documents in various formats (PDF, DOCX, TXT, etc.).
- **URL Support**: Provide a public URL to analyze documents hosted online.
- **Question Parsing**: Enter multiple questions, and the app will retrieve answers for each.
- **Supabase Integration**: Securely stores uploaded files in Supabase Storage.
- **Backend API Integration**: Communicates with a backend API for document processing and question answering.
- **Feedback Mechanism**: Collects user feedback via email (SMTP) or local CSV storage.

---

## Architecture

### High-Level Overview
LIT-RAG is built with a modular architecture to separate concerns between the frontend (Streamlit UI) and the backend (RAG system). Here's how the components interact:

1. **Frontend (Streamlit)**:
   - Handles user interactions (file uploads, URL input, question entry).
   - Sends requests to the backend API for processing.
   - Displays answers and manages feedback collection.

2. **Storage (Supabase)**:
   - Uploaded files are stored in Supabase Storage.
   - Public URLs are generated for backend access.

3. **Backend (RAG API)**:
   - Processes documents and answers questions.
   - Expects a specific JSON payload and returns answers in a predefined format.

### Workflow
1. **User Input**:
   - Users upload a file or provide a document URL.
   - Users enter one or more questions.

2. **File Handling**:
   - Uploaded files are stored in Supabase Storage.
   - A public URL is generated for the file.

3. **Backend Communication**:
   - The app sends a POST request to the backend API with the document URL and questions.
   - The backend processes the document and returns answers.

4. **Answer Display**:
   - The app displays the answers in a user-friendly format.

5. **Feedback Collection**:
   - Users can provide feedback, which is either emailed (via SMTP) or saved locally in a CSV file.

---

## Installation

### Prerequisites
- Python 3.8+
- Pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/git-tarik/lit-rag-app.git
   cd lit-rag-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up secrets:
   - Create a `.streamlit/secrets.toml` file with the following keys:
     ```toml
     [secrets]
     SUPABASE_URL = "your-supabase-url"
     SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
     BACKEND_BASE_URL = "your-backend-url"
     API_BEARER_TOKEN = "your-api-token"
     FEEDBACK_SMTP_SERVER = "smtp.example.com"
     FEEDBACK_SMTP_PORT = "587"
     FEEDBACK_SMTP_USER = "your-email@example.com"
     FEEDBACK_SMTP_PASSWORD = "your-email-password"
     FEEDBACK_TO = "feedback-recipient@example.com"
     APP_MODE = "dev"
     ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Upload a File**:
   - Click on "Upload file" and select a document.
   - Enter your questions in the text area.
   - Click "Ask" to retrieve answers.

2. **Use a URL**:
   - Switch to "Use URL" mode.
   - Enter the document's public URL.
   - Enter your questions and click "Ask".

3. **Provide Feedback**:
   - Use the feedback form to share your thoughts.

---

## Key Files
- `app.py`: Main Streamlit application.
- `requirements.txt`: Lists Python dependencies.
- `.streamlit/secrets.toml`: Stores secrets (not included in the repo).

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For questions or support, please contact [git-tarik](mailto:git-tarik@example.com).
