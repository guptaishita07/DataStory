"""
QueryMetrics - Performance tracking and analytics for SQL queries
"""

import time
import json
import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go


class QueryMetrics:
    """
    Tracks and analyzes performance metrics for SQL queries and system performance
    """
    
    def __init__(self):
        """Initialize metrics tracking"""
        if 'metrics' not in st.session_state:
            st.session_state.metrics = []
        if 'session_start' not in st.session_state:
            st.session_state.session_start = time.time()
    
    def track_query(self, nl_query: str, generated_sql: str, execution_time: float, 
                   success: bool, chart_type: str = None, data_rows: int = 0, 
                   error_message: str = None) -> None:
        """
        Track a single query execution
        
        Args:
            nl_query: Natural language query from user
            generated_sql: Generated SQL query
            execution_time: Time taken to execute (seconds)
            success: Whether query was successful
            chart_type: Type of chart generated
            data_rows: Number of rows returned
            error_message: Error message if failed
        """
        metric = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'nl_query': nl_query,
            'generated_sql': generated_sql,
            'execution_time': execution_time,
            'success': success,
            'chart_type': chart_type,
            'data_rows': data_rows,
            'error_message': error_message,
            'query_length': len(nl_query),
            'sql_length': len(generated_sql) if generated_sql else 0
        }
        
        st.session_state.metrics.append(metric)
        
        # Keep only last 100 queries to prevent memory overflow
        if len(st.session_state.metrics) > 100:
            st.session_state.metrics = st.session_state.metrics[-100:]
    
    def get_accuracy(self) -> float:
        """
        Calculate query success rate
        
        Returns:
            float: Success rate as percentage
        """
        if not st.session_state.metrics:
            return 0.0
        
        successful = sum(1 for m in st.session_state.metrics if m['success'])
        return (successful / len(st.session_state.metrics)) * 100
    
    def get_avg_response_time(self) -> float:
        """
        Calculate average response time for successful queries
        
        Returns:
            float: Average response time in seconds
        """
        if not st.session_state.metrics:
            return 0.0
        
        successful_queries = [m for m in st.session_state.metrics if m['success']]
        if not successful_queries:
            return 0.0
        
        total_time = sum(m['execution_time'] for m in successful_queries)
        return total_time / len(successful_queries)
    
    def get_total_queries(self) -> int:
        """Get total number of queries processed"""
        return len(st.session_state.metrics)
    
    def get_session_duration(self) -> float:
        """Get current session duration in minutes"""
        return (time.time() - st.session_state.session_start) / 60
    
    def get_queries_per_minute(self) -> float:
        """Calculate queries per minute rate"""
        session_duration = self.get_session_duration()
        if session_duration < 1:  # Less than 1 minute
            return 0.0
        return len(st.session_state.metrics) / session_duration
    
    def get_error_rate(self) -> float:
        """Calculate error rate as percentage"""
        if not st.session_state.metrics:
            return 0.0
        
        failed = sum(1 for m in st.session_state.metrics if not m['success'])
        return (failed / len(st.session_state.metrics)) * 100
    
    def get_chart_type_distribution(self) -> Dict[str, int]:
        """Get distribution of chart types generated"""
        chart_counts = {}
        for metric in st.session_state.metrics:
            if metric['success'] and metric['chart_type']:
                chart_type = metric['chart_type']
                chart_counts[chart_type] = chart_counts.get(chart_type, 0) + 1
        return chart_counts
    
    def get_performance_trends(self) -> pd.DataFrame:
        """Get performance trends over time"""
        if not st.session_state.metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(st.session_state.metrics)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Group by 10-minute intervals for trend analysis
        df['time_bucket'] = df['datetime'].dt.floor('10min')
        
        trends = df.groupby('time_bucket').agg({
            'execution_time': 'mean',
            'success': ['count', 'sum'],
            'data_rows': 'mean'
        }).reset_index()
        
        trends.columns = ['time', 'avg_execution_time', 'total_queries', 'successful_queries', 'avg_data_rows']
        trends['success_rate'] = (trends['successful_queries'] / trends['total_queries']) * 100
        
        return trends
    
    def get_query_complexity_analysis(self) -> Dict[str, Any]:
        """Analyze query complexity patterns"""
        if not st.session_state.metrics:
            return {}
        
        successful_queries = [m for m in st.session_state.metrics if m['success']]
        
        if not successful_queries:
            return {}
        
        # Analyze query length vs performance
        query_lengths = [m['query_length'] for m in successful_queries]
        execution_times = [m['execution_time'] for m in successful_queries]
        sql_lengths = [m['sql_length'] for m in successful_queries]
        
        # Simple correlation analysis
        length_perf_correlation = 0
        sql_perf_correlation = 0
        
        if len(query_lengths) > 1:
            length_perf_correlation = pd.Series(query_lengths).corr(pd.Series(execution_times))
            sql_perf_correlation = pd.Series(sql_lengths).corr(pd.Series(execution_times))
        
        return {
            'avg_query_length': sum(query_lengths) / len(query_lengths),
            'avg_sql_length': sum(sql_lengths) / len(successful_queries),
            'length_performance_correlation': length_perf_correlation,
            'sql_performance_correlation': sql_perf_correlation,
            'max_query_length': max(query_lengths) if query_lengths else 0,
            'min_query_length': min(query_lengths) if query_lengths else 0,
            'avg_rows_returned': sum(m['data_rows'] for m in successful_queries) / len(successful_queries)
        }
    
    def get_recent_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent error messages for debugging"""
        failed_queries = [m for m in st.session_state.metrics if not m['success']]
        # Sort by timestamp (most recent first)
        failed_queries.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return failed_queries[:limit]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'total_queries': self.get_total_queries(),
            'success_rate': self.get_accuracy(),
            'error_rate': self.get_error_rate(),
            'avg_response_time': self.get_avg_response_time(),
            'session_duration_minutes': self.get_session_duration(),
            'queries_per_minute': self.get_queries_per_minute(),
            'chart_distribution': self.get_chart_type_distribution(),
            'complexity_analysis': self.get_query_complexity_analysis()
        }
    
    def export_metrics(self) -> str:
        """Export metrics as JSON string"""
        summary = self.get_performance_summary()
        summary['raw_metrics'] = st.session_state.metrics
        summary['export_timestamp'] = datetime.now().isoformat()
        
        return json.dumps(summary, indent=2, default=str)
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics"""
        st.session_state.metrics = []
        st.session_state.session_start = time.time()
    
    def create_performance_dashboard(self) -> None:
        """Create a Streamlit dashboard showing performance metrics"""
        st.header("📊 Query Performance Dashboard")
        
        if not st.session_state.metrics:
            st.warning("No metrics data available yet. Run some queries to see performance data.")
            return
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", self.get_total_queries())
        
        with col2:
            st.metric("Success Rate", f"{self.get_accuracy():.1f}%")
        
        with col3:
            st.metric("Avg Response Time", f"{self.get_avg_response_time():.2f}s")
        
        with col4:
            st.metric("Queries/Min", f"{self.get_queries_per_minute():.1f}")
        
        # Performance trends chart
        trends_df = self.get_performance_trends()
        if not trends_df.empty:
            st.subheader("📈 Performance Trends")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(trends_df, x='time', y='avg_execution_time', 
                             title='Average Execution Time Over Time')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(trends_df, x='time', y='success_rate', 
                             title='Success Rate Over Time')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Chart type distribution
        chart_dist = self.get_chart_type_distribution()
        if chart_dist:
            st.subheader("📊 Chart Type Distribution")
            
            fig = px.pie(values=list(chart_dist.values()), 
                        names=list(chart_dist.keys()),
                        title="Distribution of Generated Chart Types")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent errors
        recent_errors = self.get_recent_errors()
        if recent_errors:
            st.subheader("⚠️ Recent Errors")
            
            for i, error in enumerate(recent_errors):
                with st.expander(f"Error {i+1}: {error['datetime'][:19]}"):
                    st.write(f"**Query:** {error['nl_query']}")
                    st.write(f"**Error:** {error['error_message']}")
                    if error['generated_sql']:
                        st.code(error['generated_sql'], language='sql')
        
        # Query complexity analysis
        complexity = self.get_query_complexity_analysis()
        if complexity:
            st.subheader("🔍 Query Complexity Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Avg Query Length", f"{complexity['avg_query_length']:.0f} chars")
                st.metric("Avg SQL Length", f"{complexity['avg_sql_length']:.0f} chars")
                st.metric("Avg Rows Returned", f"{complexity['avg_rows_returned']:.0f}")
            
            with col2:
                st.metric("Length-Performance Correlation", f"{complexity['length_performance_correlation']:.3f}")
                st.metric("SQL-Performance Correlation", f"{complexity['sql_performance_correlation']:.3f}")
        
        # Export functionality
        st.subheader("💾 Export Data")
        if st.button("Export Metrics as JSON"):
            json_data = self.export_metrics()
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"query_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        if st.button("Clear All Metrics", type="secondary"):
            if st.button("Confirm Clear", type="primary"):
                self.clear_metrics()
                st.success("Metrics cleared!")
                st.rerun()


# Usage example
def example_usage():
    """Example of how to use QueryMetrics"""
    
    # Initialize metrics tracker
    metrics = QueryMetrics()
    
    # Example: Track a query
    start_time = time.time()
    
    # Simulate query execution
    nl_query = "Show me sales data for last month"
    generated_sql = "SELECT * FROM sales WHERE date >= '2024-05-01'"
    
    # Simulate execution time
    time.sleep(0.1)
    execution_time = time.time() - start_time
    
    # Track the query
    metrics.track_query(
        nl_query=nl_query,
        generated_sql=generated_sql,
        execution_time=execution_time,
        success=True,
        chart_type="bar_chart",
        data_rows=150
    )
    
    # Display dashboard
    metrics.create_performance_dashboard()


if __name__ == "__main__":
    # For testing purposes
    example_usage()