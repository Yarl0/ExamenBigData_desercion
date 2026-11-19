import streamlit as st
import pandas as pd
import zipfile
import os
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

# ============================================================
# CONFIGURACIÓN Y RUTAS
# ============================================================
st.set_page_config(page_title="Predicción de Deserción", page_icon="🎓", layout="wide")

# Rutas relativas (funcionan en local y en la nube)
MODEL_ZIP = "model/modelo_final.zip"
EXTRACT_DIR = "model/extracted"

# ============================================================
# CARGA DEL MODELO (CONFIGURADO PARA LA NUBE)
# ============================================================
@st.cache_resource(show_spinner="Iniciando Spark y cargando el modelo...")
def cargar_modelo():
    # 1. Descomprimir el zip si es necesario
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    if not os.listdir(EXTRACT_DIR):
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    
    # 2. Iniciar Spark (Optimizado para la nube)
    spark = (SparkSession.builder
             .appName("DesercionApp")
             .master("local[*]")
             .config("spark.driver.memory", "1g")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    
    # 3. Buscar la carpeta del modelo
    model_path = None
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if "metadata" in dirs and "stages" in dirs:
            model_path = root
            break
            
    if model_path is None:
        st.error("❌ Error: No se encontró la estructura del modelo en el ZIP.")
        st.stop()

    model_path_absoluto = os.path.abspath(model_path)
    
    # 4. Cargar el modelo
    try:
        modelo = PipelineModel.load(model_path_absoluto)
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()
    
    # El orden debe ser el mismo que en tu StringIndexer
    labels = ["Dropout", "Enrolled", "Graduate"] 
    return spark, modelo, labels

spark, modelo, labels = cargar_modelo()

# ============================================================
# FORMULARIO DE USUARIO
# ============================================================
st.title("🎓 Predictor de Deserción Académica")
st.markdown("Ingresa los datos del estudiante para predecir su estado final.")

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

    # Variables fijas
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

    submit = st.form_submit_button("🔍 Predecir Estado", use_container_width=True)

# ============================================================
# LÓGICA DE PREDICCIÓN
# ============================================================
if submit:
    # 1. Diccionario de datos
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

    # 2. Pandas a Spark
    df_pd = pd.DataFrame([datos_crudos])
    
    # Forzamos que las categóricas sean String para que el StringIndexer no falle
    columnas_string = ["marital_status", "application_mode", "application_order", "course", 
                       "daytime_evening_attendance", "previous_qualification", "nacionality", 
                       "mother_s_qualification", "father_s_qualification", "mother_s_occupation", 
                       "father_s_occupation", "displaced", "educational_special_needs", "debtor", 
                       "tuition_fees_up_to_date", "gender", "scholarship_holder", "international"]
    
    for col in columnas_string:
        df_pd[col] = df_pd[col].astype(str)
        
    df_spark = spark.createDataFrame(df_pd)

    # 3. Variables derivadas (En la app, como pide el taller)
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

    # 4. Predecir y mostrar
    try:
        prediccion = modelo.transform(df_spark)
        
        resultado = prediccion.select(
            "prediction", 
            vector_to_array("probability").alias("probs")
        ).collect()[0]
        
        clase_predicha_idx = int(resultado["prediction"])
        probabilidades = resultado["probs"]
        clase_texto = labels[clase_predicha_idx]
        
        st.divider()
        if clase_texto == "Dropout":
            st.error("### ⚠️ Resultado: Riesgo de Deserción (Dropout)")
            st.write("El modelo detecta alta probabilidad de abandono. Se recomienda intervención.")
        elif clase_texto == "Enrolled":
            st.warning("### ⏳ Resultado: Matriculado (Enrolled)")
            st.write("El estudiante permanecerá matriculado, pero requiere monitoreo.")
        else:
            st.success("### 🎓 Resultado: Graduado (Graduate)")
            st.write("El estudiante tiene alta probabilidad de completar sus estudios exitosamente.")

        st.write("**Probabilidades por clase:**")
        for i, label in enumerate(labels):
            st.progress(float(probabilidades[i]), text=f"{label}: {probabilidades[i]*100:.1f}%")

    except Exception as e:
        st.error("Ocurrió un error al predecir.")
        st.exception(e)