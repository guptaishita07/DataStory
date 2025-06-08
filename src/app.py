from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import hashlib
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========================
# METRICS TRACKING CLASS
# ========================
class QueryMetrics:
    def __init__(self):
        if 'metrics' not in st.session_state:
            st.session_state.metrics = []
    
    def track_query(self, nl_query: str, generated_sql: str, execution_time: float, success: bool, chart_type: str = None):
        metric = {
            'timestamp': time.time(),
            'nl_query': nl_query,
            'generated_sql': generated_sql,
            'execution_time': execution_time,
            'success': success,
            'chart_type': chart_type
        }
        st.session_state.metrics.append(metric)
    
    def get_accuracy(self) -> float:
        if not st.session_state.metrics:
            return 0.0
        successful = sum(1 for m in st.session_state.metrics if m['success'])
        return (successful / len(st.session_state.metrics)) * 100
    
    def get_avg_response_time(self) -> float:
        if not st.session_state.metrics:
            return 0.0
        times = [m['execution_time'] for m in st.session_state.metrics if m['success']]
        return sum(times) / len(times) if times else 0.0

# ========================
# VISUALIZATION ENGINE
# ========================
class DataVisualizer:
    def __init__(self):
        self.chart_types = {
            'time_series': ['line', 'area'],
            'categorical': ['bar', 'pie', 'donut'],
            'numerical': ['scatter', 'histogram', 'box'],
            'comparison': ['bar', 'column', 'radar']
        }
    
    def detect_chart_type(self, df: pd.DataFrame, query: str) -> str:
        """Intelligently detect the best chart type based on data and query"""
        if df.empty:
            return 'table'
        
        # Check for time-series keywords
        time_keywords = ['time', 'date', 'month', 'year', 'day', 'trend', 'over time']
        if any(keyword in query.lower() for keyword in time_keywords):
            return 'line'
        
        # Check for comparison keywords
        comparison_keywords = ['top', 'bottom', 'highest', 'lowest', 'compare', 'vs', 'versus']
        if any(keyword in query.lower() for keyword in comparison_keywords):
            return 'bar'
        
        # Check for distribution keywords
        distribution_keywords = ['distribution', 'spread', 'range', 'histogram']
        if any(keyword in query.lower() for keyword in distribution_keywords):
            return 'histogram'
        
        # Auto-detect based on data structure
        if len(df.columns) == 2:
            if df.dtypes.iloc[1] in ['int64', 'float64']:
                return 'bar'
        elif len(df.columns) > 2:
            return 'scatter'
        
        return 'table'
    
    def create_visualization(self, df: pd.DataFrame, query: str, chart_type: str = None) -> go.Figure:
        """Create appropriate visualization based on data and query"""
        if df.empty:
            return self.create_empty_chart()
        
        if chart_type is None:
            chart_type = self.detect_chart_type(df, query)
        
        try:
            if chart_type == 'line':
                return self.create_line_chart(df)
            elif chart_type == 'bar':
                return self.create_bar_chart(df)
            elif chart_type == 'pie':
                return self.create_pie_chart(df)
            elif chart_type == 'scatter':
                return self.create_scatter_plot(df)
            elif chart_type == 'histogram':
                return self.create_histogram(df)
            else:
                return self.create_bar_chart(df)  # Default fallback
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            return self.create_empty_chart()
    
    def create_line_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create line chart for time-series data"""
        fig = px.line(df, x=df.columns[0], y=df.columns[1], 
                     title=f"{df.columns[1]} Over {df.columns[0]}")
        fig.update_layout(template="plotly_white", height=500)
        return fig
    
    def create_bar_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create bar chart for categorical data"""
        # Take top 10 to avoid overcrowding
        if len(df) > 10:
            df = df.head(10)
        
        fig = px.bar(df, x=df.columns[0], y=df.columns[1],
                    title=f"{df.columns[1]} by {df.columns[0]}")
        fig.update_layout(template="plotly_white", height=500)
        return fig
    
    def create_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create pie chart for categorical data"""
        # Take top 8 to avoid overcrowding
        if len(df) > 8:
            df = df.head(8)
        
        fig = px.pie(df, values=df.columns[1], names=df.columns[0],
                    title=f"Distribution of {df.columns[1]}")
        fig.update_layout(height=500)
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create scatter plot for numerical relationships"""
        if len(df.columns) >= 2:
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1],
                           title=f"{df.columns[1]} vs {df.columns[0]}")
            fig.update_layout(template="plotly_white", height=500)
            return fig
        return self.create_bar_chart(df)
    
    def create_histogram(self, df: pd.DataFrame) -> go.Figure:
        """Create histogram for distribution analysis"""
        numeric_col = df.select_dtypes(include=[np.number]).columns[0]
        fig = px.histogram(df, x=numeric_col, 
                          title=f"Distribution of {numeric_col}")
        fig.update_layout(template="plotly_white", height=500)
        return fig
    
    def create_empty_chart(self) -> go.Figure:
        """Create empty chart for error cases"""
        fig = go.Figure()
        fig.add_annotation(text="No data to visualize", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", height=300)
        return fig

# ========================
# INSIGHTS GENERATOR
# ========================
class InsightsGenerator:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_insights(self, df: pd.DataFrame, query: str, sql_query: str) -> str:
        """Generate natural language insights from query results"""
        if df.empty:
            return "No data found for your query."
        
        # Basic statistics
        stats = self.get_basic_stats(df)
        
        # Create insights prompt
        prompt = f"""
        As a data analyst, provide key insights from this query result:
        
        Original Question: {query}
        SQL Query: {sql_query}
        
        Data Summary:
        - Rows: {len(df)}
        - Columns: {list(df.columns)}
        {stats}
        
        Sample Data:
        {df.head().to_string()}
        
        Provide 2-3 key insights in bullet points. Be specific and actionable.
        Focus on trends, patterns, or notable findings.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def get_basic_stats(self, df: pd.DataFrame) -> str:
        """Get basic statistics about the dataframe"""
        stats = []
        
        # Numeric columns stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                stats.append(f"- {col}: Mean={df[col].mean():.2f}, Max={df[col].max()}, Min={df[col].min()}")
        
        # Categorical columns stats
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                unique_count = df[col].nunique()
                stats.append(f"- {col}: {unique_count} unique values")
        
        return "\n".join(stats) if stats else "No notable statistics available"

# ========================
# PREDICTION ENGINE
# ========================
class PredictionEngine:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
    
    def can_predict(self, df: pd.DataFrame, query: str) -> bool:
        """Check if data is suitable for prediction"""
        prediction_keywords = ['predict', 'forecast', 'future', 'trend', 'next']
        if not any(keyword in query.lower() for keyword in prediction_keywords):
            return False
        
        # Need at least 5 rows and 2 columns with numeric data
        if len(df) < 5 or len(df.columns) < 2:
            return False
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return len(numeric_cols) >= 2
    
    def simple_forecast(self, df: pd.DataFrame, periods: int = 3) -> Dict[str, Any]:
        """Simple linear regression forecast"""
        try:
            # Assume first column is index/time, second is target
            X = np.arange(len(df)).reshape(-1, 1)
            y = df.iloc[:, 1].values
            
            # Fit model
            self.model.fit(X, y)
            
            # Predict future values
            future_X = np.arange(len(df), len(df) + periods).reshape(-1, 1)
            predictions = self.model.predict(future_X)
            
            return {
                'predictions': predictions.tolist(),
                'trend': 'increasing' if predictions[-1] > predictions[0] else 'decreasing',
                'confidence': 'low'  # Simple model, low confidence
            }
        except Exception as e:
            return {'error': str(e)}

# ========================
# ENHANCED DATABASE FUNCTIONS
# ========================
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    # llm = ChatOpenAI(model="gpt-4-0125-preview")
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_enhanced_response(user_query: str, db: SQLDatabase, chat_history: list):
    """Enhanced response with visualizations and insights"""
    start_time = time.time()
    
    try:
        # Generate SQL
        sql_chain = get_sql_chain(db)
        sql_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
        
        # Execute SQL and get results
        results = db.run(sql_query)
        
        # Convert to DataFrame for visualization
        # This is a simplified conversion - you might need to adjust based on your data structure
        if results and results != "[]":
            # Parse results into DataFrame
            df = pd.read_sql(sql_query, db._engine)
        else:
            df = pd.DataFrame()
        
        # Initialize components
        visualizer = DataVisualizer()
        llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
        insights_gen = InsightsGenerator(llm)
        predictor = PredictionEngine()
        
        # Create visualization
        chart = visualizer.create_visualization(df, user_query)
        chart_type = visualizer.detect_chart_type(df, user_query)
        
        # Generate insights
        insights = insights_gen.generate_insights(df, user_query, sql_query)
        
        # Check for predictions
        predictions = None
        if predictor.can_predict(df, user_query):
            predictions = predictor.simple_forecast(df)
        
        # Generate natural language response
        template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        
        <SCHEMA>{schema}</SCHEMA>
        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
        
        Keep the response concise but informative. The user will also see a visualization and detailed insights separately.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            RunnablePassthrough.assign(query=lambda _: sql_query).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda _: results,
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
        
        # Track metrics
        execution_time = time.time() - start_time
        metrics = QueryMetrics()
        metrics.track_query(user_query, sql_query, execution_time, True, chart_type)
        
        return {
            'response': response,
            'chart': chart,
            'insights': insights,
            'predictions': predictions,
            'sql_query': sql_query,
            'execution_time': execution_time,
            'data_rows': len(df)
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        metrics = QueryMetrics()
        metrics.track_query(user_query, "", execution_time, False)
        
        return {
            'response': f"Error processing query: {str(e)}",
            'chart': None,
            'insights': None,
            'predictions': None,
            'sql_query': None,
            'execution_time': execution_time,
            'data_rows': 0
        }

# ========================
# STREAMLIT UI
# ========================
def display_metrics():
    """Display performance metrics in sidebar"""
    metrics = QueryMetrics()
    st.sidebar.subheader("📊 Performance Metrics")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Accuracy", f"{metrics.get_accuracy():.1f}%")
    with col2:
        st.metric("Avg Response", f"{metrics.get_avg_response_time():.2f}s")
    
    if st.session_state.get('metrics'):
        st.sidebar.text(f"Total Queries: {len(st.session_state.metrics)}")

# ========================
# MAIN APPLICATION
# ========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm DataStoryteller AI. Ask me anything about your database and I'll provide insights with visualizations."),
    ]

load_dotenv()

st.set_page_config(page_title="DataStoryteller AI", page_icon="📊", layout="wide")

st.title("📊 DataStoryteller AI")
st.markdown("*Beyond Queries to Insights & Visualizations*")

with st.sidebar:
    st.subheader("🔧 Database Settings")
    st.write("Connect to your database and start exploring data with AI-powered insights.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="admin", key="Password")
    st.text_input("Database", value="Chinook", key="Database")
    
    if st.button("🔗 Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("✅ Connected to database!")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
    
    # Display metrics
    display_metrics()

# Chat interface
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Ask me about your data...")
if user_query is not None and user_query.strip() != "" and 'db' in st.session_state:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        with st.spinner("Analyzing your data..."):
            result = get_enhanced_response(user_query, st.session_state.db, st.session_state.chat_history)
        
        # Display main response
        st.markdown(result['response'])
        
        # Display visualization if available
        if result['chart']:
            st.plotly_chart(result['chart'], use_container_width=True)
        
        # Display insights
        if result['insights']:
            st.subheader("🔍 Key Insights")
            st.markdown(result['insights'])
        
        # Display predictions if available
        if result['predictions'] and 'predictions' in result['predictions']:
            st.subheader("📈 Predictions")
            pred_data = result['predictions']
            st.write(f"**Trend**: {pred_data.get('trend', 'Unknown')}")
            st.write(f"**Next 3 values**: {pred_data['predictions']}")
            st.caption("*Predictions are based on simple linear regression and should be used as rough estimates.*")
        
        # Show query details in expander
        with st.expander("🔧 Query Details"):
            st.code(result['sql_query'], language="sql")
            st.caption(f"Executed in {result['execution_time']:.2f}s | Returned {result['data_rows']} rows")
        
    st.session_state.chat_history.append(AIMessage(content=result['response']))