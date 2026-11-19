import streamlit as st
import pandas as pd
import zipfile
import os
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import functions as F
import shutil

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(page_title="Predicción de Deserción Académica", page_icon="🎓", layout="wide")

MODEL_ZIP = "model/modelo_final.zip"
EXTRACT_DIR = "model/extracted"

# ============================================================
# CARGA DEL MODELO (Punto A del taller)
# ============================================================
@st.cache_resource(show_spinner="Iniciando Spark y cargando el modelo...")
def cargar_modelo():
    # 1. Descomprimir el zip si no está descomprimido
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    if not os.listdir(EXTRACT_DIR):
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    
    # 2. Iniciar Spark (configurando memoria para la nube)
    spark = (SparkSession.builder
             .appName("DesercionApp")
             .master("local[*]")
             .config("spark.driver.memory", "1g") # Límite de memoria para evitar caídas en Streamlit Cloud
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    
    # 3. Cargar el modelo
    # Buscamos la carpeta real que contiene el modelo (por si el zip creó subcarpetas)
    model_path = None
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if "metadata" in dirs and "stages" in dirs:
            model_path = root
            break
            
    modelo = PipelineModel.load(model_path)
    
    # Definimos las etiquetas (deben estar en el mismo orden que en Colab)
    # Si en tu Colab el orden fue distinto, cámbialo aquí.
    labels = ["Dropout", "Enrolled", "Graduate"] 
    
    return spark, modelo, labels

spark, modelo, labels = cargar_modelo()

# ============================================================
# INTERFAZ DE USUARIO (Punto B del taller)
# ============================================================
st.title("🎓 Predictor de Deserción Académica")
st.markdown("Ingresa los datos del estudiante para predecir su estado final (Dropout, Enrolled o Graduate).")

with st.form("formulario_estudiante"):
    st.subheader("1. Datos Demográficos y Socioeconómicos")
    col1, col2, col3 = st.columns(3)
    with col1:
        marital_status = st.selectbox("Estado civil", options=[1, 2, 3, 4, 5, 6])
        gender = st.selectbox("Género", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
        age = st.number_input("Edad al ingresar", min_value=17, max_value=70, value=20)
    with col2:
        displaced = st.selectbox("Desplazado", options=[0, 1], format_func=lambda x: "No" if x==0 else "Sí")
        debtor = st.selectbox("Deudor", options=[0, 1], format_func=lambda x: "No" if x==0 else "Sí")
        tuition = st.selectbox("Matrícula al día", options=[0, 1], format_func=lambda x: "No" if x==0 else "Sí")
    with col3:
        scholarship = st.selectbox("Becado", options=[0, 1], format_func=lambda x: "No" if x==0 else "Sí")
        international = st.selectbox("Internacional", options=[0, 1], format_func=lambda x: "No" if x==0 else "Sí")

    st.subheader("2. Datos Académicos (Semestre 1)")
    c1, c2, c3 = st.columns(3)
    with c1:
        cred_1 = st.number_input("Créditos 1er sem", min_value=0, value=0)
        enroll_1 = st.number_input("Inscritos 1er sem", min_value=0, value=6)
    with c2:
        eval_1 = st.number_input("Evaluaciones 1er sem", min_value=0, value=8)
        approve_1 = st.number_input("Aprobados 1er sem", min_value=0, value=5)
    with c3:
        grade_1 = st.number_input("Nota 1er sem", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        without_eval_1 = st.number_input("Sin evaluación 1er sem", min_value=0, value=0)

    st.subheader("3. Datos Académicos (Semestre 2)")
    c4, c5, c6 = st.columns(3)
    with c4:
        cred_2 = st.number_input("Créditos 2do sem", min_value=0, value=0)
        enroll_2 = st.number_input("Inscritos 2do sem", min_value=0, value=6)
    with c5:
        eval_2 = st.number_input("Evaluaciones 2do sem", min_value=0, value=8)
        approve_2 = st.number_input("Aprobados 2do sem", min_value=0, value=5)
    with c6:
        grade_2 = st.number_input("Nota 2do sem", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        without_eval_2 = st.number_input("Sin evaluación 2do sem", min_value=0, value=0)

    # Variables fijas para simplificar el formulario (puedes agregarlas si quieres)
    course = 9500
    app_mode = 1
    app_order = 1
    daytime = 1
    prev_qual = 1
    nacionality = 1
    mother_qual = 1
    father_qual = 1
    mother_occ = 1
    father_occ = 1
    special_needs = 0
    unemployment = 10.8
    inflation = 1.4
    gdp = 1.74

    submit = st.form_submit_button("🔍 Predecir Estado del Estudiante")

# ============================================================
# PREDICCIÓN (Puntos C, D y E del taller)
# ============================================================
if submit:
    # 1. Crear un diccionario con los datos crudos
    datos_crudos = {
        "marital_status": marital_status, "application_mode": app_mode, "application_order": app_order,
        "course": course, "daytime_evening_attendance": daytime, "previous_qualification": prev_qual,
        "nacionality": nacionality, "mother_s_qualification": mother_qual, "father_s_qualification": father_qual,
        "mother_s_occupation": mother_occ, "father_s_occupation": father_occ, "displaced": displaced,
        "educational_special_needs": special_needs, "debtor": debtor, "tuition_fees_up_to_date": tuition,
        "gender": gender, "scholarship_holder": scholarship, "age_at_enrollment": age,
        "international": international, 
        "curricular_units_1st_sem__credited": cred_1, "curricular_units_1st_sem__enrolled": enroll_1,
        "curricular_units_1st_sem__evaluations": eval_1, "curricular_units_1st_sem__approved": approve_1,
        "curricular_units_1st_sem__grade": grade_1, "curricular_units_1st_sem__without_evaluations": without_eval_1,
        "curricular_units_2nd_sem__credited": cred_2, "curricular_units_2nd_sem__enrolled": enroll_2,
        "curricular_units_2nd_sem__evaluations": eval_2, "curricular_units_2nd_sem__approved": approve_2,
        "curricular_units_2nd_sem__grade": grade_2, "curricular_units_2nd_sem__without_evaluations": without_eval_2,
        "unemployment_rate": unemployment, "inflation_rate": inflation, "gdp": gdp
    }

    # 2. Convertir a DataFrame de Pandas y luego a Spark
    # Nota: Spark infiere los tipos automáticamente, pero forzamos los numéricos a double
    df_pd = pd.DataFrame([datos_crudos])
    # Convertimos todo a float para evitar conflictos de tipos con Spark
    df_pd = df_pd.astype(float) 
    
    df_spark = spark.createDataFrame(df_pd)

    # 3. Calcular las variables derivadas EN LA APP (Punto C)
    df_spark = df_spark.withColumn(
        "tasa_aprobacion_1s",
        F.when(F.col("curricular_units_1st_sem__enrolled") > 0, 
               F.col("curricular_units_1st_sem__approved") / F.col("curricular_units_1st_sem__enrolled")).otherwise(0.0)
    ).withColumn(
        "tasa_aprobacion_2s",
        F.when(F.col("curricular_units_2nd_sem__enrolled") > 0, 
               F.col("curricular_units_2nd_sem__approved") / F.col("curricular_units_2nd_sem__enrolled")).otherwise(0.0)
    ).withColumn(
        "variacion_nota",
        F.col("curricular_units_2nd_sem__grade") - F.col("curricular_units_1st_sem__grade")
    )

    # 4. Aplicar el modelo (Punto D)
    prediccion = modelo.transform(df_spark)
    
    # 5. Extraer resultados (Punto E)
    # Convertimos el vector de probabilidad a un array de Python
    from pyspark.ml.functions import vector_to_array
    resultado = prediccion.select(
        "prediction", 
        vector_to_array("probability").alias("probs")
    ).collect()[0]
    
    clase_predicha_idx = int(resultado["prediction"])
    probabilidades = resultado["probs"]
    
    # Traducir usando la lista de labels
    clase_texto = labels[clase_predicha_idx]
    
    # 6. Mostrar resultados en la UI
    st.divider()
    if clase_texto == "Dropout":
        st.error(f"### ⚠️ Resultado: Riesgo de Deserción (Dropout)")
        st.write("El modelo detecta una alta probabilidad de que el estudiante abandone sus estudios. Se recomienda intervención inmediata.")
    elif clase_texto == "Enrolled":
        st.warning(f"### ⏳ Resultado: Matriculado (Enrolled)")
        st.write("El estudiante permanecerá matriculado, pero se debe monitorear su progreso.")
    else:
        st.success(f"### 🎓 Resultado: Graduado (Graduate)")
        st.write("El estudiante tiene alta probabilidad de completar sus estudios exitosamente.")

    # Mostrar probabilidades detalladas
    st.write("**Probabilidades por clase:**")
    for i, label in enumerate(labels):
        st.progress(float(probabilidades[i]), text=f"{label}: {probabilidades[i]*100:.1f}%")
