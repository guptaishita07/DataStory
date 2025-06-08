"""
InsightsGenerator - AI-powered natural language insights from query results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


class InsightsGenerator:
    """
    Generates natural language insights and summaries from SQL query results using LLM
    """
    
    def __init__(self, llm):
        """
        Initialize with language model
        
        Args:
            llm: Language model instance (ChatOpenAI or ChatGroq)
        """
        self.llm = llm
        self.insight_templates = {
            'summary': self._create_summary_template(),
            'trends': self._create_trends_template(),
            'comparisons': self._create_comparison_template(),
            'anomalies': self._create_anomaly_template()
        }
    
    def generate_insights(self, df: pd.DataFrame, query: str, sql_query: str, 
                         insight_type: str = 'summary') -> str:
        """
        Generate natural language insights from query results
        
        Args:
            df: DataFrame containing query results
            query: Original natural language query
            sql_query: Generated SQL query
            insight_type: Type of insights to generate
            
        Returns:
            str: Natural language insights
        """
        if df.empty:
            return "❌ No data found for your query. Please try rephrasing your question or check if the data exists."
        
        try:
            # Get data statistics and patterns
            stats = self.get_comprehensive_stats(df)
            patterns = self.detect_patterns(df, query)
            
            # Select appropriate template
            template = self.insight_templates.get(insight_type, self.insight_templates['summary'])
            
            # Create insights prompt
            insights_prompt = template.format(
                original_query=query,
                sql_query=sql_query,
                row_count=len(df),
                columns=list(df.columns),
                data_stats=stats,
                patterns=patterns,
                sample_data=self.format_sample_data(df),
                data_types=self.get_data_types_summary(df)
            )
            
            # Generate insights using LLM
            response = self.llm.invoke(insights_prompt)
            insights = response.content if hasattr(response, 'content') else str(response)
            
            # Add data quality indicators
            quality_notes = self.get_data_quality_notes(df)
            if quality_notes:
                insights += f"\n\n**Data Quality Notes:**\n{quality_notes}"
            
            return insights
            
        except Exception as e:
            return f"⚠️ Error generating insights: {str(e)}\n\nHowever, your query returned {len(df)} rows of data."
    
    def get_comprehensive_stats(self, df: pd.DataFrame) -> str:
        """Get comprehensive statistics about the dataframe"""
        stats = []
        
        # Overall data info
        stats.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats.append("\n**Numeric Data:**")
            for col in numeric_cols:
                col_stats = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'max': df[col].max(),
                    'min': df[col].min(),
                    'std': df[col].std()
                }
                stats.append(f"- {col}: Mean={col_stats['mean']:.2f}, Median={col_stats['median']:.2f}, Range={col_stats['min']:.2f}-{col_stats['max']:.2f}")
        
        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            stats.append("\n**Categorical Data:**")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                most_common = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                stats.append(f"- {col}: {unique_count} unique values, most common: '{most_common}'")
        
        # Missing data
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            stats.append(f"\n**Missing Data:** {missing_data.sum()} total missing values")
        
        return "\n".join(stats)
    
    def detect_patterns(self, df: pd.DataFrame, query: str) -> str:
        """Detect patterns and trends in the data"""
        patterns = []
        
        # Time series patterns
        if self._has_time_series_data(df):
            time_patterns = self._analyze_time_series(df)
            if time_patterns:
                patterns.append(f"Time Series: {time_patterns}")
        
        # Distribution patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    skew_direction = "right" if skewness > 0 else "left"
                    patterns.append(f"{col} is heavily skewed {skew_direction}")
        
        # Correlation patterns
        if len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()
            high_corr = []
            for i in range(len(correlations.columns)):
                for j in range(i+1, len(correlations.columns)):
                    corr_val = correlations.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        col1, col2 = correlations.columns[i], correlations.columns[j]
                        high_corr.append(f"{col1} and {col2} are {'positively' if corr_val > 0 else 'negatively'} correlated ({corr_val:.2f})")
            if high_corr:
                patterns.extend(high_corr)
        
        # Outlier detection
        outliers = self._detect_outliers(df)
        if outliers:
            patterns.extend(outliers)
        
        return "\n".join(patterns) if patterns else "No significant patterns detected"
    
    def get_data_quality_notes(self, df: pd.DataFrame) -> str:
        """Generate data quality assessment"""
        notes = []
        
        # Check for missing values
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 5:
            notes.append(f"⚠️ {missing_pct:.1f}% of data is missing")
        elif missing_pct > 0:
            notes.append(f"ℹ️ {missing_pct:.1f}% of data is missing (minimal)")
        
        # Check for duplicates
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            notes.append(f"⚠️ {duplicate_rows} duplicate rows found")
        
        # Check data freshness (if date columns exist)
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            latest_date = df[date_cols[0]].max()
            notes.append(f"📅 Latest data: {latest_date}")
        
        return "\n".join(notes)
    
    def format_sample_data(self, df: pd.DataFrame, max_rows: int = 5) -> str:
        """Format sample data for LLM prompt"""
        sample = df.head(max_rows)
        return sample.to_string(index=False, max_cols=10)
    
    def get_data_types_summary(self, df: pd.DataFrame) -> str:
        """Get summary of data types"""
        type_counts = df.dtypes.value_counts()
        return f"Data types: {dict(type_counts)}"
    
    def _has_time_series_data(self, df: pd.DataFrame) -> bool:
        """Check if dataframe contains time series data"""
        date_cols = df.select_dtypes(include=['datetime64']).columns
        return len(date_cols) > 0 or any('date' in col.lower() or 'time' in col.lower() for col in df.columns)
    
    def _analyze_time_series(self, df: pd.DataFrame) -> str:
        """Analyze time series patterns"""
        # Simple trend analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            first_val = df[numeric_cols[0]].iloc[0]
            last_val = df[numeric_cols[0]].iloc[-1]
            if last_val > first_val * 1.1:
                return "Increasing trend detected"
            elif last_val < first_val * 0.9:
                return "Decreasing trend detected"
            else:
                return "Stable trend"
        return ""
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[str]:
        """Simple outlier detection using IQR method"""
        outliers = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            
            if outlier_count > 0:
                outliers.append(f"{col} has {outlier_count} potential outliers")
        
        return outliers
    
    def _create_summary_template(self) -> str:
        """Create template for general summary insights"""
        return """
        As a senior data analyst, provide key insights from this database query result:
        
        **Original Question:** {original_query}
        **SQL Query:** {sql_query}
        
        **Data Overview:**
        - Rows returned: {row_count}
        - Columns: {columns}
        - Data types: {data_types}
        
        **Statistics:**
        {data_stats}
        
        **Patterns Detected:**
        {patterns}
        
        **Sample Data:**
        {sample_data}
        
        Please provide 3-4 key insights in bullet points:
        - Focus on the most important findings
        - Highlight any surprising or notable trends
        - Include actionable recommendations when possible
        - Use business-friendly language
        - Include relevant numbers and percentages
        
        Format your response with clear bullet points and emojis for visual appeal.
        """
    
    def _create_trends_template(self) -> str:
        """Create template for trend analysis"""
        return """
        Analyze trends and patterns in this data:
        
        **Query:** {original_query}
        **Data:** {row_count} rows, {columns}
        
        **Statistics:** {data_stats}
        **Patterns:** {patterns}
        **Sample:** {sample_data}
        
        Focus on:
        - Trending patterns (increasing, decreasing, seasonal)
        - Growth rates and changes over time
        - Comparative analysis between categories
        - Future implications based on current trends
        
        Provide trend insights with specific numbers and percentages.
        """
    
    def _create_comparison_template(self) -> str:
        """Create template for comparison analysis"""
        return """
        Provide comparative analysis of this data:
        
        **Query:** {original_query}
        **Results:** {row_count} records
        
        **Data Summary:** {data_stats}
        **Patterns:** {patterns}
        
        Focus on:
        - Top performers vs bottom performers
        - Significant differences between categories
        - Performance gaps and opportunities
        - Ranking and relative positioning
        
        Highlight the most significant comparisons with concrete numbers.
        """
    
    def _create_anomaly_template(self) -> str:
        """Create template for anomaly detection"""
        return """
        Identify anomalies and unusual patterns in this data:
        
        **Query:** {original_query}
        **Data:** {data_stats}
        **Patterns:** {patterns}
        
        Look for:
        - Outliers and unusual values
        - Unexpected patterns or breaks in trends
        - Data quality issues
        - Anomalous behavior worth investigating
        
        Explain what makes these findings unusual and their potential implications.
        """