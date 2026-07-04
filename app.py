# demo/app.py
import streamlit as st
import torch
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import json
from datetime import datetime
from inference import TrafficSignInference
from models.model_factory import ModelFactory
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка страницы
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚗",
    layout="wide"
)

class DemoApp:
    def __init__(self):
        self.model = None
        self.history = []
        self.load_model()
        self.load_class_names()
    
    def load_model(self):
        """Загрузка лучшей модели"""
        model_name = 'efficientnet_b0'  # Изменяем на лучшую модель
        model_interface = ModelFactory.create_model(model_name, 43, (224, 224))
        
        model_path = f'checkpoints/best_model_{model_name}.pth'
        if os.path.exists(model_path):
            self.model = TrafficSignInference(model_path, model_interface)
            st.sidebar.success(f"Model {model_name} loaded successfully!")
        else:
            st.sidebar.warning("Model not found. Using demo mode.")
            self.model = None
    
    def load_class_names(self):
        """Загрузка названий классов"""
        try:
            with open('data/class_names.json', 'r') as f:
                self.class_names = json.load(f)
        except:
            self.class_names = None
    
    def predict_image(self, image):
        """Предсказание для загруженного изображения"""
        if self.model is None:
            return None, None, None
        
        # Сохранение временного файла
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        try:
            predicted_class, class_name, confidence = self.model.predict(temp_path)
            top_k = self.model.get_top_k_predictions(temp_path, 5)
            
            # Сохранение в историю
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'predicted_class': predicted_class,
                'class_name': class_name,
                'confidence': confidence,
                'top_k': top_k
            })
            
            os.remove(temp_path)
            return predicted_class, class_name, confidence, top_k
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None, None, None, None
    
    def show_statistics(self):
        """Отображение статистики"""
        if self.history:
            df = pd.DataFrame(self.history)
            
            # Общая статистика
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(df))
            with col2:
                avg_conf = df['confidence'].mean() if 'confidence' in df else 0
                st.metric("Average Confidence", f"{avg_conf:.2%}")
            with col3:
                # Самый частый класс
                if 'class_name' in df:
                    most_common = df['class_name'].mode().iloc[0] if not df.empty else 'N/A'
                    st.metric("Most Common Sign", most_common)
            
            # График уверенности
            if 'confidence' in df:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df['confidence'], nbinsx=20))
                fig.update_layout(
                    title="Confidence Distribution",
                    xaxis_title="Confidence",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def show_demo(self):
        """Главная страница демонстрации"""
        st.title("🚗 Traffic Sign Recognition System")
        st.markdown("""
        Загрузите изображение дорожного знака, и система определит его класс 
        с указанием уверенности.
        """)
        
        # Загрузка изображения
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Отображение загруженного изображения
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Предсказание
            with st.spinner("Analyzing..."):
                result = self.predict_image(image)
                if result:
                    predicted_class, class_name, confidence, top_k = result
                    
                    with col2:
                        st.markdown("### Prediction Results")
                        st.markdown(f"**Class:** {class_name}")
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        st.markdown(f"**Class ID:** {predicted_class}")
                        
                        # Индикатор уверенности
                        st.progress(confidence)
                        
                        # Топ-5 предсказаний
                        st.markdown("### Top 5 Predictions")
                        fig = go.Figure()
                        
                        top_labels = [f"{name} ({conf:.2%})" for _, name, conf in top_k]
                        top_values = [conf for _, _, conf in top_k]
                        
                        fig.add_trace(go.Bar(
                            x=top_values,
                            y=top_labels,
                            orientation='h',
                            text=[f"{v:.2%}" for v in top_values],
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=20, b=20),
                            xaxis_title="Confidence"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        # История и статистика
        st.markdown("---")
        st.markdown("## 📊 Session Statistics")
        self.show_statistics()
        
        # Кнопка сброса истории
        if st.button("Clear History"):
            self.history = []
            st.success("History cleared!")
    
    def show_model_comparison(self):
        """Страница сравнения моделей"""
        st.title("📊 Model Comparison")
        
        # Загрузка результатов сравнения
        if os.path.exists('reports/results.json'):
            with open('reports/results.json', 'r') as f:
                results = json.load(f)
            
            # Создание DataFrame
            df = pd.DataFrame(results).T
            df = df.reset_index().rename(columns={'index': 'Model'})
            
            # Метрики для отображения
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            # Сравнение метрик
            fig = go.Figure()
            for metric in metrics:
                if metric in df.columns:
                    fig.add_trace(go.Bar(
                        name=metric.capitalize(),
                        x=df['Model'],
                        y=df[metric],
                        text=[f"{v:.3f}" for v in df[metric]],
                        textposition='auto'
                    ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                barmode='group',
                xaxis_title="Model",
                yaxis_title="Score",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Таблица с результатами
            st.markdown("### Detailed Results")
            st.dataframe(df[['Model'] + metrics].round(4))
            
            # Визуализация размера модели и скорости
            col1, col2 = st.columns(2)
            
            with col1:
                if 'model_size_mb' in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df['Model'],
                        y=df['model_size_mb'],
                        text=[f"{v:.1f}M" for v in df['model_size_mb']],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title="Model Size (Parameters)",
                        xaxis_title="Model",
                        yaxis_title="Size (M)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'avg_inference_time_ms' in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df['Model'],
                        y=df['avg_inference_time_ms'],
                        text=[f"{v:.1f}ms" for v in df['avg_inference_time_ms']],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title="Inference Time",
                        xaxis_title="Model",
                        yaxis_title="Time (ms)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No comparison results found. Please run the training script first.")
    
    def show_error_analysis(self):
        """Анализ ошибок"""
        st.title("🔍 Error Analysis")
        
        # Загрузка изображения для анализа ошибок
        uploaded_file = st.file_uploader(
            "Upload an image for error analysis...",
            type=['jpg', 'jpeg', 'png'],
            key="error_analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Ввод правильного класса
            true_class = st.number_input(
                "Enter the correct class ID:",
                min_value=0,
                max_value=42,
                value=0,
                step=1
            )
            
            if st.button("Analyze"):
                with st.spinner("Analyzing errors..."):
                    # Сохранение временного файла
                    temp_path = "temp_analysis.jpg"
                    image.save(temp_path)
                    
                    try:
                        if self.model:
                            analysis = self.model.analyze_errors(temp_path, true_class)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(image, caption="Analyzed Image", use_column_width=True)
                                
                            with col2:
                                st.markdown("### Analysis Results")
                                if analysis['is_correct']:
                                    st.success("✅ Prediction is correct!")
                                else:
                                    st.error("❌ Prediction is wrong!")
                                
                                st.markdown(f"**True class:** {analysis['true_class_name']}")
                                st.markdown(f"**Predicted class:** {analysis['predicted_class_name']}")
                                st.markdown(f"**Confidence:** {analysis['confidence']:.2%}")
                            
                            # Топ-3 предсказания
                            st.markdown("### Top 3 Predictions")
                            top_df = pd.DataFrame(
                                analysis['top_3_predictions'],
                                columns=['Class ID', 'Class Name', 'Confidence']
                            )
                            st.dataframe(top_df)
                            
                            os.remove(temp_path)
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

def main():
    app = DemoApp()
    
    # Сайдбар
    st.sidebar.title("🚦 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🖼️ Demo", "📊 Model Comparison", "🔍 Error Analysis"]
    )
    
    # Отображение выбранной страницы
    if page == "🖼️ Demo":
        app.show_demo()
    elif page == "📊 Model Comparison":
        app.show_model_comparison()
    elif page == "🔍 Error Analysis":
        app.show_error_analysis()

if __name__ == "__main__":
    main()