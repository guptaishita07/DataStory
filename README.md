
# 📊 DataStory AI: Beyond Queries to Insights & Visualizations

**DataStory AI** is an intelligent data exploration tool that empowers users to interact with their databases using natural language. Built with **Streamlit**, **LangChain**, and **Plotly**, this application not only translates natural language questions into SQL queries but also visualizes the results and generates actionable insights, complete with predictive capabilities. It transforms complex database interactions into intuitive conversations.

---

## ✨ Features

- **Natural Language to SQL**: Ask questions in plain English, and DataStoryteller AI will convert them into precise SQL queries.
- **Dynamic Data Visualization**: Automatically generates interactive charts (line, bar, pie, scatter, histogram) based on query results, intelligently detecting the most suitable chart type.
- **AI-Powered Insights**: Get key insights and trends extracted from your data, presented in an easy-to-understand format.
- **Simple Trend Prediction**: Offers basic linear regression-based predictions for suitable datasets.
- **Interactive Chat Interface**: A user-friendly Streamlit interface for seamless, conversational interaction.
- **Configurable Database Connection**: Easily connect to your MySQL database via the sidebar settings.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A MySQL Database (or any database supported by `SQLAlchemy` and `langchain_community.utilities.SQLDatabase`)
## API Keys Setup

To run the **DataStoryteller AI** app, you may need API keys depending on which models or services you want to use. Below are the details and how to obtain them.

---

### 1. OpenAI API Key (optional)

The OpenAI API key is used if you want to enable the `ChatOpenAI` model integration.

**How to get your OpenAI API key:**

1. Visit the [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. Log in or create an account.
3. Create a new secret key and copy it.

**Set the key:**

Add the key to your `.env` file as:

```env
OPENAI_API_KEY=your_openai_api_key_here

```
### 2. Groq API Key (required for ChatGroq)

The Groq API key is **required** if you want to use the ChatGroq integration in your app.

**How to get your Groq API key:**

1. Visit the [Groq Developer Portal](https://developer.groq.com/)
2. Sign up or log in.
3. Navigate to the **API Keys** section.
4. Generate and copy your API key.

**Set the key:**

Add the key to your `.env` file as:

```env
GROQ_API_KEY=your_groq_api_key_here
```
### Example `.env` file

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=grq-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DATABASE_URL=postgresql://user:password@host:port/dbname
```


> 💡 Recommendation: Use the [Chinook Database](https://github.com/lerocha/chinook-database) for testing queries.

---

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/datastoryteller-ai.git
cd datastoryteller-ai
````

### Set Up Project Structure

Ensure your local project directory matches this structure:

```
datastoryteller-ai/
├── src/
│   ├── app.py              # Main Streamlit app orchestrating UI and flow
│   ├── __init__.py         # Makes 'src' a package for relative imports
│   ├── visualizer.py       # DataVisualizer class for Plotly charts
│   ├── insights.py         # InsightsGenerator and PredictionEngine classes
│   └── metrics.py          # QueryMetrics class for tracking SQL query performance
├── requirements.txt        # Required Python libraries
├── .env                    # API and DB credentials
└── README.md               # Project documentation

```

### Create `.env` File

In the root directory, create a `.env` file:

```
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration & Database Setup

### Database Connection

1. Ensure your MySQL server is running.
2. Launch the app and open the sidebar.
3. Input your:

   * Host
   * Port
   * Username
   * Password
   * Database name
4. Click **🔗 Connect** to establish the connection.

### API Keys

Ensure your `.env` file contains valid keys for OpenAI and/or Groq.

---

## ▶️ Running the Application

To start the application:

```bash
streamlit run src/app.py
```

Open your browser and go to `http://localhost:8501` if it doesn’t open automatically.

---

## 🏗️ Project Structure

```
datastoryteller-ai/
├── src/
│   ├── app.py              # Main Streamlit app orchestrating UI and flow
│   ├── __init__.py         # Makes 'src' a package for relative imports
│   ├── visualizer.py       # DataVisualizer class for Plotly charts
│   └── insights.py         # InsightsGenerator and PredictionEngine classes
├── requirements.txt        # Required Python libraries
├── .env                    # API and DB credentials
└── README.md               # Project documentation
```

---

## 💡 How it Works

1. **User Input**: Type a natural language question in the chat.
2. **SQL Generation**: LangChain translates it to a SQL query using schema info.
3. **Query Execution**: Executes SQL against your connected database.
4. **Data Processing**: Converts raw results into a Pandas DataFrame.
5. **Visualization**: Automatically selects and renders the best chart using Plotly.
6. **Insight Generation**: AI summarizes key insights using LLMs.
7. **Trend Prediction (Optional)**: Uses linear regression for simple forecasts.
8. **Natural Language Summary**: Provides a concise explanation of the result.

---

## 📌 Technologies Used

* **Streamlit** – Interactive UI
* **LangChain** – LLM chaining and SQL generation
* **Plotly** – Data visualizations
* **Pandas** – Data handling
* **Groq / OpenAI** – Natural language understanding and insight generation

---
