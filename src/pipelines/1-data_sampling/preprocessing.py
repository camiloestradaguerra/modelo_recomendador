import os
import pandas as pd
import boto3
import s3fs
from pathlib import Path
import re
import json
#from skimpy import clean_columns
from datetime import datetime


class DataPreprocessingPipeline:
    """Data preprocessing pipeline for recommendation system"""

    def __init__(self):

        self.aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.environ.get("AWS_DEFAULT_REGION")

        if not all([self.aws_access_key, self.aws_secret_key, self.aws_region]):
            raise ValueError("Faltan variables de entorno AWS en el archivo .env")
        
        self.bucket_name = 'dcelip-dev-artifacts-s3'
        self.file_raw_path = 'mlops/input/raw/'
        self.file_procesed_path = 'mlops/input/processed/'
        self.file_name_socios = 'df_socios_concat_20251114_065013.parquet' #'df_socios_concat.parquet'
        self.file_name_establecimientos = 'df_establecimientos_concat_20251114_065013.parquet' #'df_establecimientos_concat.parquet'
        self.file_name_entrenamiento = 'df_entrenamiento_concat_20251114_065014.parquet' #'df_entrenamiento_concat.parquet'

    def _load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw data from parquet files in S3"""

        print("Configurando acceso a S3...")
        s3 = boto3.client('s3')
        fs = s3fs.S3FileSystem()

        s3_uri_socios = f's3://{self.bucket_name}/{self.file_raw_path + self.file_name_socios}'
        s3_uri_establecimientos = f's3://{self.bucket_name}/{self.file_raw_path + self.file_name_establecimientos}'
        s3_uri_entrenamiento = f's3://{self.bucket_name}/{self.file_raw_path + self.file_name_entrenamiento}'

        print("Leyendo archivos desde S3...")

        df_socios = pd.read_parquet(
            s3_uri_socios,
            storage_options={'key': self.aws_access_key, 'secret': self.aws_secret_key}
        )
        df_establecimientos = pd.read_parquet(
            s3_uri_establecimientos,
            storage_options={'key': self.aws_access_key, 'secret': self.aws_secret_key}
        )
        df_entrenamiento = pd.read_parquet(
            s3_uri_entrenamiento,
            storage_options={'key': self.aws_access_key, 'secret': self.aws_secret_key}
        )

        print("Archivos cargados correctamente. ")

        # df_socios = df_socios[:100]
        # df_establecimientos = df_establecimientos[:100]
        # df_entrenamiento = df_entrenamiento[:100]

        return df_socios, df_establecimientos, df_entrenamiento

    def data_extended(self) -> pd.DataFrame:
        """Cleans and merges socios, establecimientos, and entrenamiento datasets"""
        print("Iniciando proceso de limpieza...")
        df_socios, df_establecimientos, df_entrenamiento = self._load_raw_data()

        # --- Limpieza socios ---
        df_socios = df_socios[['Id_Persona', 'ESTADO_CIVIL', 'Edad', 'GENERO', 'ROL',
                            'Antiguedad_Socio_Unico', 'SEGMENTO_COMERCIAL', 'Ciudad',
                            'Zona', 'Region']]
        df_socios['Id_Persona'] = pd.to_numeric(df_socios['Id_Persona'], errors='coerce')
        df_socios['Antiguedad_Socio_Unico'] = df_socios['Antiguedad_Socio_Unico'].fillna(
            df_socios['Antiguedad_Socio_Unico'].mean()
        )

        for col in ['Ciudad', 'Zona', 'Region']:
            moda = df_socios[col].mode(dropna=True)[0]
            df_socios[col] = df_socios[col].fillna(moda)

        # --- Limpieza establecimientos ---
        df_establecimientos = df_establecimientos[['ID_ESTABLECIMIENTO', 'CADENA', 'ESTABLECIMIENTO']]
        df_establecimientos['ID_ESTABLECIMIENTO'] = pd.to_numeric(df_establecimientos['ID_ESTABLECIMIENTO'], errors="coerce")

        # --- Limpieza entrenamiento ---
        df_entrenamiento = df_entrenamiento[['DiaID', 'Id_Persona', 'ID_ESTABLECIMIENTO',
                                            'ESPECIALIDAD', 'HORA_INICIO', 'HORA_FIN',
                                            'LOCALIZACION_EXTERNA', 'MONTO', 'Neteo_Mensual', 'Neteo_Diario']]
        df_entrenamiento['Id_Persona'] = pd.to_numeric(df_entrenamiento['Id_Persona'], errors='coerce')
        df_entrenamiento['ID_ESTABLECIMIENTO'] = pd.to_numeric(df_entrenamiento['ID_ESTABLECIMIENTO'], errors="coerce")
        df_entrenamiento['MONTO'] = pd.to_numeric(df_entrenamiento['MONTO'], errors='coerce')
        df_entrenamiento['DiaID'] = pd.to_datetime(df_entrenamiento['DiaID'], errors='coerce')
        df_entrenamiento['HORA_INICIO'] = pd.to_datetime(
            df_entrenamiento['DiaID'].dt.strftime('%Y-%m-%d') + ' ' + df_entrenamiento['HORA_INICIO'],
            errors='coerce'
        )
        df_entrenamiento['HORA_FIN'] = pd.to_datetime(
            df_entrenamiento['DiaID'].dt.strftime('%Y-%m-%d') + ' ' + df_entrenamiento['HORA_FIN'],
            errors='coerce'
        )

        df_entrenamiento = df_entrenamiento[df_entrenamiento["Id_Persona"].isin(df_socios["Id_Persona"])]

        df_merge1 = df_entrenamiento.merge(df_socios, on='Id_Persona', how='left')
        data_extendida_clean = (
            df_merge1.merge(df_establecimientos, on='ID_ESTABLECIMIENTO', how='left')
            .dropna(subset=['CADENA', 'ESTABLECIMIENTO', 'ESPECIALIDAD', 'LOCALIZACION_EXTERNA'])
        )

        return data_extendida_clean
    
    def clean_recent_text_columns(self, df) -> pd.DataFrame:
            
        df.columns = df.columns.str.lower()
            
        columnas_object = df.select_dtypes(include='object').columns
        df[columnas_object] = df[columnas_object].astype(str)   

        for col in columnas_object:
            df[col] = df[col].astype(str).apply(lambda texto: re.sub("\s\s+", " ", 
                        re.sub(r"[\r\n]+", ' ', 
                        re.sub('\[#*.>=\]', '', 
                        re.sub('[^a-z ]', ' ', 
                        re.sub(" \d+", ' ', 
                        texto.lower()))))).strip())
        
        df['id_persona'] = df['id_persona'].astype(float)
        df['id_persona'] = df['id_persona'].astype(int)
        df['antiguedad_socio_unico'] = df['antiguedad_socio_unico'].astype(int)
        df['especialidad'] = df['especialidad'].str.strip().str.title()
        df['localizacion_externa'] = df['localizacion_externa'].str.strip().str.title()
        df['estado_civil'] = df['estado_civil'].str.strip().str.title()
        df['genero'] = df['genero'].str.strip().str.title()
        df['segmento_comercial'] = df['segmento_comercial'].str.strip().str.title()
        df['ciudad'] =  df['ciudad'].str.strip().str.title()
        df['zona'] = df['zona'].str.strip().str.title()
        df['region'] = df['region'].str.strip().str.title()
        df['cadena'] = df['cadena'].str.strip().str.title()
        df['establecimiento'] = df['establecimiento'].str.strip().str.title()

        return df
    
    def outliers_filters(self, df) -> pd.DataFrame:

        df = df[df['antiguedad_socio_unico']<df['antiguedad_socio_unico'].quantile(0.99)].reset_index(drop=True)

        df_by_city = df.groupby('ciudad')['antiguedad_socio_unico'].size().reset_index().sort_values(by='antiguedad_socio_unico', ascending=False)

        ciudades_con_alta_antiguedad_df = df_by_city[df_by_city['antiguedad_socio_unico'] > 25]
        nombres_de_ciudades = ciudades_con_alta_antiguedad_df['ciudad'].unique()

        df_filtrado = df[df['ciudad'].isin(nombres_de_ciudades)]

        df_frec_cadena = df_filtrado['cadena'].value_counts(ascending=False).reset_index()

        filtro_cadena = df_frec_cadena[df_frec_cadena['count'] < 7]['cadena'].unique()
        
        df_filtrado_cadena = df_filtrado[~df_filtrado['cadena'].isin(filtro_cadena)]

        return df_filtrado_cadena
    
    def normalization_establecimientos(self, df) -> pd.DataFrame:

        if not all([self.aws_access_key, self.aws_secret_key, self.aws_region]):
            raise EnvironmentError(
                "Faltan variables de entorno AWS. Verifica que AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY y AWS_DEFAULT_REGION estén definidos."
        )


        print("Configurando acceso a S3...")
        s3 = boto3.client('s3')
        fs = s3fs.S3FileSystem()
        
        diccionario_establecimientos = {}

        with open("diccionario_completo.txt", "r", encoding="utf-8") as f:
            for linea in f:
                if ":" in linea:
                    clave, valor = linea.strip().split(":", 1)
                    diccionario_establecimientos[clave.strip()] = valor.strip()


        df['establecimiento'] = df['establecimiento'].astype(str).str.strip().str.lower()

        df['establecimiento'] = df['establecimiento'].map(diccionario_establecimientos).fillna(df['establecimiento'])

        df_establecimiento = df['establecimiento'].value_counts(ascending=False).reset_index()

        filtro_establecimiento = df_establecimiento[df_establecimiento['count'] < 7]['establecimiento'].unique()
        
        df_filtrado_establecimiento = df[~df['establecimiento'].isin(filtro_establecimiento)]

        data_extendida_clean = df_filtrado_establecimiento.copy()

        # ---------------------------------------------------------
        # Guardar resultado procesado en S3
        # ---------------------------------------------------------
        # s3_uri_output = f's3://{self.bucket_name}/{self.file_procesed_path + "data_extendida_clean3.parquet"}'

        # print(f"Guardando archivo procesado en: {s3_uri_output}")

        # data_extendida_clean.to_parquet(
        #     s3_uri_output,
        #     engine='pyarrow',
        #     storage_options={
        #         'key': self.aws_access_key,
        #         'secret': self.aws_secret_key
        #     }
        # )

        # print("Archivo cargado exitosamente en S3.")

        return data_extendida_clean
    
    def save_bucket_data(self, path_destino, nombre_archivo, df):
        """Guarda un DataFrame como parquet en el bucket destino con timestamp."""

        # ===== NUEVA LÓGICA =====
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Dividir nombre y extensión
        base, ext = nombre_archivo.split(".")
        nombre_archivo_timestamp = f"{base}_{timestamp}.{ext}"
        # =========================

        ruta_relativa = f"{path_destino}{nombre_archivo_timestamp}"
        ruta_s3_destino = f"s3://{self.bucket_dest}/{ruta_relativa}"

        try:
            with self.fs.open(ruta_s3_destino, 'wb') as f:
                df.to_parquet(f, index=False)
            print(f"Archivo guardado correctamente en: {ruta_s3_destino}")
        except Exception as e:
            print(f"Error al guardar en S3: {e}")

# ---------------------------------------------------------
# Bloque de prueba
# ---------------------------------------------------------
# if __name__ == "__main__":
#     pipeline = DataPreprocessingPipeline()
#     data_extendida = pipeline.data_extended()
#     data_extendida_clean = pipeline.clean_recent_text_columns(data_extendida)
#     data_extendida_clean_outliers = pipeline.outliers_filters(data_extendida_clean)
#     data_extendida_clean_outliers_normalizada = pipeline.normalization_establecimientos(data_extendida_clean_outliers)
#     print("\n Pipeline ejecutado correctamente.")
#     print(f"Shape final: {data_extendida_clean_outliers.shape}")
#     print(f"Columnas: {list(data_extendida_clean_outliers.columns)}")
#     print(data_extendida_clean_outliers.head(5))
if __name__ == "__main__":
    pipeline = DataPreprocessingPipeline()
    data_extendida = pipeline.data_extended()
    # Normalización
    #data_extendida_clean_outliers_normalizada = pipeline.normalization_establecimientos(data_extendida_clean_outliers)
    
    # Guardado en S3
    pipeline.save_bucket_data(
        path_destino="mlops/input/processed/",  # solo el path relativo dentro del bucket
        nombre_archivo="data_extendida_ensayo.parquet",
        df=data_extendida
    )