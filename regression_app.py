import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import io

# Функция для вычисления метрик
def calculate_metrics(y_true, y_pred):
    metrics = {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }
    # MAPE с обработкой нулевых значений
    mask = y_true != 0
    if np.any(mask):
        metrics["MAPE"] = 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    else:
        metrics["MAPE"] = float('nan')
    return metrics

# Функция для построения регрессии
def build_regression(model_name, X_train, y_train, X_test):
    if model_name == "Линейная":
        model = LinearRegression()
    elif model_name == "Полиномиальная (степень=2)":
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    elif model_name == "Гребневая (Ridge)":
        model = Ridge(alpha=1.0)
    elif model_name == "Лассо (Lasso)":
        model = Lasso(alpha=0.1)
    elif model_name == "Хьюбер (Huber)":
        model = HuberRegressor()
    elif model_name == "RANSAC":
        model = RANSACRegressor()
    elif model_name == "Сплайн (кубический)":
        spline = UnivariateSpline(X_train.flatten(), y_train, k=3)
        return spline(X_test.flatten())
    elif model_name == "Нейросеть (MLP)":
        model = MLPRegressor(hidden_layer_sizes=(50, 20), max_iter=1000, random_state=42)
    
    model.fit(X_train, y_train)
    return model.predict(X_test)

# Настройка страницы Streamlit
st.set_page_config(page_title="Анализ регрессии", layout="wide")
st.title("📈 Анализ регрессии с отрицательными значениями")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл Excel", type=["xlsx", "xls"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Файл успешно загружен!")
        
        # Выбор столбцов
        cols = df.columns.tolist()
        col1, col2 = st.columns(2)
        
        with col1:
            target = st.selectbox("Выберите целевую переменную (Y):", cols)
        with col2:
            feature = st.selectbox("Выберите признак (X):", cols)
        
        # Подготовка данных
        X = df[[feature]].values
        y = df[target].values
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Выбор модели
        models = [
            "Линейная",
            "Полиномиальная (степень=2)",
            "Гребневая (Ridge)",
            "Лассо (Lasso)",
            "Хьюбер (Huber)",
            "RANSAC",
            "Сплайн (кубический)",
            "Нейросеть (MLP)"
        ]
        selected_model = st.selectbox("Выберите тип регрессии:", models)
        
        # Кнопка для запуска анализа
        if st.button("Построить регрессию"):
            with st.spinner("Идет построение модели..."):
                # Построение модели
                y_pred = build_regression(selected_model, X_train, y_train, X_test)
                
                # Вычисление метрик
                metrics = calculate_metrics(y_test, y_pred)
                
                # Отображение метрик
                st.subheader("📊 Метрики модели")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("R²", f"{metrics['R2']:.3f}")
                col2.metric("MSE", f"{metrics['MSE']:.3f}")
                col3.metric("RMSE", f"{metrics['RMSE']:.3f}")
                col4.metric("MAE", f"{metrics['MAE']:.3f}")
                col5.metric("MAPE", f"{metrics.get('MAPE', 'N/A'):.3f}%")
                
                # Визуализация
                st.subheader("📈 График регрессии")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Точки данных
                ax.scatter(X, y, color='blue', alpha=0.5, label='Данные')
                
                # Линия регрессии
                x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
                if selected_model == "Сплайн (кубический)":
                    spline = UnivariateSpline(X_train.flatten(), y_train, k=3)
                    y_plot = spline(x_plot.flatten())
                else:
                    model = None
                    if selected_model == "Линейная":
                        model = LinearRegression()
                    elif selected_model == "Полиномиальная (степень=2)":
                        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                    # ... (аналогично для других моделей)
                    
                    if model is not None:
                        model.fit(X_train, y_train)
                        y_plot = model.predict(x_plot)
                
                ax.plot(x_plot, y_plot, color='red', linewidth=2, label=selected_model)
                ax.set_xlabel(feature)
                ax.set_ylabel(target)
                ax.legend()
                ax.grid(True)
                
                st.pyplot(fig)
                
                # Вывод коэффициентов для линейных моделей
                if selected_model in ["Линейная", "Гребневая (Ridge)", "Лассо (Lasso)"]:
                    st.subheader("🔍 Коэффициенты модели")
                    if hasattr(model, 'coef_'):
                        coef_df = pd.DataFrame({
                            "Признак": [feature],
                            "Коэффициент": [model.coef_[0]],
                            "Intercept": [model.intercept_]
                        })
                        st.dataframe(coef_df)
    
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")
else:
    st.info("Пожалуйста, загрузите файл Excel для начала анализа.")

# Инструкция в боковой панели
with st.sidebar:
    st.markdown("## Инструкция")
    st.markdown("""
    1. Загрузите файл Excel с данными
    2. Выберите целевую переменную (Y)
    3. Выберите признак (X)
    4. Выберите тип регрессии
    5. Нажмите "Построить регрессию"
    """)
    st.markdown("## Поддерживаемые типы регрессии")
    st.markdown("""
    - Линейная
    - Полиномиальная (степень 2)
    - Гребневая (Ridge)
    - Лассо (Lasso)
    - Хьюбер (Huber)
    - RANSAC
    - Сплайн (кубический)
    - Нейросеть (MLP)
    """)

if __name__ == "__main__":
    pass
