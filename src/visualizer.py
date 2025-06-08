"""
DataVisualizer - Automated chart generation for SQL query results
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st


class DataVisualizer:
    """
    Automatically generates appropriate visualizations based on data structure and query context
    """
    
    def __init__(self):
        self.chart_types = {
            'time_series': ['line', 'area'],
            'categorical': ['bar', 'pie', 'donut'],
            'numerical': ['scatter', 'histogram', 'box'],
            'comparison': ['bar', 'column', 'radar']
        }
    
    def detect_chart_type(self, df: pd.DataFrame, query: str) -> str:
        """
        Intelligently detect the best chart type based on data structure and query keywords
        
        Args:
            df: DataFrame containing query results
            query: Original natural language query
            
        Returns:
            str: Recommended chart type
        """
        if df.empty:
            return 'table'
        
        # Check for time-series keywords
        time_keywords = ['time', 'date', 'month', 'year', 'day', 'trend', 'over time', 'timeline']
        if any(keyword in query.lower() for keyword in time_keywords):
            return 'line'
        
        # Check for comparison keywords
        comparison_keywords = ['top', 'bottom', 'highest', 'lowest', 'compare', 'vs', 'versus', 'best', 'worst']
        if any(keyword in query.lower() for keyword in comparison_keywords):
            return 'bar'
        
        # Check for distribution keywords
        distribution_keywords = ['distribution', 'spread', 'range', 'histogram', 'frequency']
        if any(keyword in query.lower() for keyword in distribution_keywords):
            return 'histogram'
        
        # Check for proportion keywords
        proportion_keywords = ['percentage', 'proportion', 'share', 'breakdown', 'composition']
        if any(keyword in query.lower() for keyword in proportion_keywords):
            return 'pie'
        
        # Auto-detect based on data structure
        if len(df.columns) == 2:
            # Check if first column looks like dates
            if df.dtypes.iloc[0] in ['datetime64[ns]', 'object']:
                first_col_sample = str(df.iloc[0, 0]).lower()
                if any(date_indicator in first_col_sample for date_indicator in ['2023', '2024', '2025', 'jan', 'feb', 'mar']):
                    return 'line'
            
            # Check if second column is numeric
            if df.dtypes.iloc[1] in ['int64', 'float64']:
                return 'bar'
        elif len(df.columns) > 2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                return 'scatter'
        
        return 'table'
    
    def create_visualization(self, df: pd.DataFrame, query: str, chart_type: str = None) -> go.Figure:
        """
        Create appropriate visualization based on data and query
        
        Args:
            df: DataFrame containing query results
            query: Original natural language query
            chart_type: Optional override for chart type
            
        Returns:
            plotly.graph_objects.Figure: Interactive chart
        """
        if df.empty:
            return self.create_empty_chart("No data to visualize")
        
        if chart_type is None:
            chart_type = self.detect_chart_type(df, query)
        
        try:
            if chart_type == 'line':
                return self.create_line_chart(df, query)
            elif chart_type == 'bar':
                return self.create_bar_chart(df, query)
            elif chart_type == 'pie':
                return self.create_pie_chart(df, query)
            elif chart_type == 'scatter':
                return self.create_scatter_plot(df, query)
            elif chart_type == 'histogram':
                return self.create_histogram(df, query)
            else:
                return self.create_bar_chart(df, query)  # Default fallback
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            return self.create_empty_chart(f"Error creating chart: {str(e)}")
    
    def create_line_chart(self, df: pd.DataFrame, query: str) -> go.Figure:
        """Create line chart for time-series data"""
        x_col, y_col = df.columns[0], df.columns[1]
        
        fig = px.line(df, x=x_col, y=y_col, 
                     title=f"{y_col} Over {x_col}",
                     markers=True)
        
        fig.update_layout(
            template="plotly_white", 
            height=500,
            hovermode='x unified',
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
    
    def create_bar_chart(self, df: pd.DataFrame, query: str) -> go.Figure:
        """Create bar chart for categorical data"""
        # Take top 15 to avoid overcrowding
        if len(df) > 15:
            df = df.head(15)
            title_suffix = " (Top 15)"
        else:
            title_suffix = ""
        
        x_col, y_col = df.columns[0], df.columns[1]
        
        fig = px.bar(df, x=x_col, y=y_col,
                    title=f"{y_col} by {x_col}{title_suffix}",
                    color=y_col,
                    color_continuous_scale='viridis')
        
        fig.update_layout(
            template="plotly_white", 
            height=500,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            showlegend=False
        )
        
        # Rotate x-axis labels if they're long
        if df[x_col].astype(str).str.len().max() > 10:
            fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_pie_chart(self, df: pd.DataFrame, query: str) -> go.Figure:
        """Create pie chart for categorical data"""
        # Take top 8 to avoid overcrowding
        if len(df) > 8:
            top_df = df.head(7)
            other_sum = df.iloc[7:][df.columns[1]].sum()
            other_row = pd.DataFrame({df.columns[0]: ['Others'], df.columns[1]: [other_sum]})
            df = pd.concat([top_df, other_row], ignore_index=True)
        
        values_col, names_col = df.columns[1], df.columns[0]
        
        fig = px.pie(df, values=values_col, names=names_col,
                    title=f"Distribution of {values_col}",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_layout(height=500)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, query: str) -> go.Figure:
        """Create scatter plot for numerical relationships"""
        if len(df.columns) >= 2:
            x_col, y_col = df.columns[0], df.columns[1]
            
            # Use third column for color if available
            color_col = df.columns[2] if len(df.columns) > 2 else None
            
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=f"{y_col} vs {x_col}",
                           trendline="ols" if len(df) > 3 else None)
            
            fig.update_layout(
                template="plotly_white", 
                height=500,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title()
            )
            
            return fig
        return self.create_bar_chart(df, query)
    
    def create_histogram(self, df: pd.DataFrame, query: str) -> go.Figure:
        """Create histogram for distribution analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return self.create_bar_chart(df, query)
        
        numeric_col = numeric_cols[0]
        
        fig = px.histogram(df, x=numeric_col, 
                          title=f"Distribution of {numeric_col}",
                          nbins=min(20, len(df)//2) if len(df) > 10 else 10)
        
        fig.update_layout(
            template="plotly_white", 
            height=500,
            xaxis_title=numeric_col.replace('_', ' ').title(),
            yaxis_title="Frequency"
        )
        
        return fig
    
    def create_empty_chart(self, message: str = "No data to visualize") -> go.Figure:
        """Create empty chart for error cases"""
        fig = go.Figure()
        fig.add_annotation(
            text=message, 
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template="plotly_white", 
            height=300,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    def get_chart_recommendations(self, df: pd.DataFrame, query: str) -> List[str]:
        """
        Get list of recommended chart types for given data
        
        Returns:
            List of recommended chart types
        """
        recommendations = []
        
        if len(df.columns) >= 2:
            recommendations.append('bar')
            
        if len(df.columns) == 2 and df.dtypes.iloc[1] in ['int64', 'float64']:
            recommendations.extend(['line', 'pie'])
            
        if len(df.columns) > 2:
            recommendations.append('scatter')
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            recommendations.append('histogram')
            
        return list(set(recommendations))  # Remove duplicates