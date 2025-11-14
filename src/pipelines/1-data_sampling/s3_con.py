import os
import re
import sys
import boto3
import pandas as pd
import s3fs
from datetime import datetime
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv


# ------------------------------------------------------------
#                  CONFIGURACIÓN DEL LOGGER
# ------------------------------------------------------------
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
    level="INFO"
)
log_path = Path("logs/pipeline.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logger.add(
    log_path,
    rotation="1 day",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="INFO",
    enqueue=True
)

# ------------------------------------------------------------
#                    S3 DATA MANAGER
# ------------------------------------------------------------
class S3DataManager:
    def __init__(self):
        self.fs = None
        self._load_env_credentials()
        self._init_s3()

    def _load_env_credentials(self):
        """Carga credenciales AWS desde .env o sistema."""
        load_dotenv()
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION')

        if not all([self.aws_access_key, self.aws_secret_key, self.aws_region]):
            raise ValueError("Faltan variables de entorno AWS (.env o sistema).")

        logger.info("Credenciales AWS cargadas correctamente.")

    def _init_s3(self):
        """Inicializa cliente boto3 y S3FileSystem."""
        boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.aws_region
        )
        self.fs = s3fs.S3FileSystem(
            key=self.aws_access_key,
            secret=self.aws_secret_key,
            client_kwargs={'region_name': self.aws_region}
        )
        logger.info("Conexión S3 inicializada correctamente.")

    def load_dataframe_from_s3(self, bucket: str, prefix: str, limit: int = None) -> pd.DataFrame:
        """Carga y concatena archivos Parquet desde S3 con un prefijo."""
        path_s3 = f"{bucket}/{prefix}"
        
        parquet_files = [
            f"s3://{file}"
            for file in self.fs.ls(path_s3)
            if file.endswith('.parquet')
        ]

        logger.info(f"Archivos encontrados en {prefix}: {len(parquet_files)}")

        if not parquet_files:
            logger.warning(f"No hay archivos parquet en {prefix}.")
            return pd.DataFrame()

        df_list = []
        for i, file in enumerate(parquet_files[0:2]):
            if limit and i >= limit:
                break
            df = pd.read_parquet(
                file,
                storage_options={
                    'key': self.aws_access_key,
                    'secret': self.aws_secret_key
                }
            )
            df_list.append(df)

        df_concat = pd.concat(df_list, ignore_index=True)
        logger.info(f"Registros concatenados: {df_concat.shape[0]}")

        return df_concat
    
    def load_single_parquet(self, s3_uri: str) -> pd.DataFrame:
        """
        Carga un único archivo parquet desde un S3 URI exacto.
        Ejemplo de s3_uri: 's3://bucket/path/file.parquet'
        """
        df = pd.read_parquet(
            s3_uri,
            storage_options={
                "key": self.aws_access_key,
                "secret": self.aws_secret_key
            }
        )
        return df


    def save_dataframe_to_s3(self, df: pd.DataFrame, bucket: str, path_destino: str, nombre_archivo: str):
        """Guarda un DataFrame en S3 con timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = nombre_archivo.split(".")
        nombre_archivo_timestamp = f"{base}_{timestamp}.{ext}"

        ruta_s3_destino = f"s3://{bucket}/{path_destino}{nombre_archivo_timestamp}"

        try:
            with self.fs.open(ruta_s3_destino, 'wb') as f:
                df.to_parquet(f, index=False)
            logger.success(f"Archivo guardado correctamente: {ruta_s3_destino}")
        except Exception as e:
            logger.error(f"Error guardando en S3: {e}")
            raise
    
    def get_newest_file_by_date(self, bucket_name: str, prefix: str = "", starts_with: str = ""):
        """
        Retorna la ruta s3://bucket/key del archivo más reciente dentro de un prefijo S3.
        Filtra por archivos que comiencen con `starts_with` si se proporciona.
        La búsqueda de la fecha se hace por el filename (YYYY-MM-DD, YYYYMMDD, YYYYMMDD_HHMMSS)
        Si no encuentra fecha en filenames, usa LastModified como fallback.
        """
        logger.info(f"Buscando archivo más reciente en s3://{bucket_name}/{prefix} con inicio '{starts_with}'")

        s3 = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.aws_region
        )

        date_pattern = r"(\d{4}[-_]?\d{2}[-_]?\d{2})(?:[_-]?(\d{6}))?"

        paginator = s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        newest_file = None
        newest_date = None

        fallback_file = None
        fallback_date = None

        total_checked = 0
        for page in page_iterator:
            for obj in page.get("Contents", []):
                total_checked += 1
                key = obj["Key"]

                # Filtrar por el comienzo del nombre si se indica
                filename = key.split("/")[-1]
                if starts_with and not filename.startswith(starts_with):
                    continue

                # 1) intentar extraer fecha del nombre
                m = re.search(date_pattern, key)
                if m:
                    date_part = m.group(1).replace("_", "-")
                    time_part = m.group(2)

                    parsed = None
                    for fmt in ("%Y-%m-%d", "%Y%m%d"):
                        try:
                            parsed = datetime.strptime(date_part, fmt)
                            break
                        except Exception:
                            pass

                    if parsed and time_part:
                        try:
                            t = datetime.strptime(time_part, "%H%M%S").time()
                            parsed = datetime.combine(parsed.date(), t)
                        except Exception:
                            pass

                    if parsed:
                        if (newest_date is None) or (parsed > newest_date):
                            newest_date = parsed
                            newest_file = key

                # fallback por LastModified
                lm = obj.get("LastModified")
                if lm:
                    if (fallback_date is None) or (lm > fallback_date):
                        fallback_date = lm
                        fallback_file = key

        logger.info(f"Se revisaron {total_checked} objetos en S3 bajo el prefijo.")

        if newest_file:
            ruta = f"s3://{bucket_name}/{newest_file}"
            logger.success(f"Archivo más reciente encontrado por nombre: {ruta}")
            return ruta

        if fallback_file:
            ruta = f"s3://{bucket_name}/{fallback_file}"
            logger.warning("No se encontró fecha en nombres; retornando el archivo más reciente por LastModified.")
            logger.success(f"Archivo más reciente (fallback): {ruta}")
            return ruta

        logger.warning("No se encontró ningún archivo con fecha válida ni objetos en el prefijo.")
        return None

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
        """Load raw data from the newest parquet files in S3 using S3DataManager."""

        print("Configurando acceso a S3...")

        s3_manager = S3DataManager()  # instancia de S3DataManager ya configurada con credenciales

        # Obtener la ruta del archivo más reciente para cada dataset
        s3_uri_socios = s3_manager.get_newest_file_by_date(
            bucket_name=self.bucket_name,
            prefix=self.file_raw_path,
            starts_with="df_socios"
        )

        s3_uri_establecimientos = s3_manager.get_newest_file_by_date(
            bucket_name=self.bucket_name,
            prefix=self.file_raw_path,
            starts_with="df_establecimientos"
        )

        s3_uri_entrenamiento = s3_manager.get_newest_file_by_date(
            bucket_name=self.bucket_name,
            prefix=self.file_raw_path,
            starts_with="df_entrenamiento"
        )

        if not all([s3_uri_socios, s3_uri_establecimientos, s3_uri_entrenamiento]):
            raise FileNotFoundError("No se encontraron archivos recientes para todos los datasets.")

        print("Leyendo archivos más recientes desde S3...")

        df_socios = pd.read_parquet(
            s3_uri_socios,
            storage_options={'key': s3_manager.aws_access_key, 'secret': s3_manager.aws_secret_key}
        )

        df_establecimientos = pd.read_parquet(
            s3_uri_establecimientos,
            storage_options={'key': s3_manager.aws_access_key, 'secret': s3_manager.aws_secret_key}
        )

        df_entrenamiento = pd.read_parquet(
            s3_uri_entrenamiento,
            storage_options={'key': s3_manager.aws_access_key, 'secret': s3_manager.aws_secret_key}
        )

        print("Archivos cargados correctamente:")
        print(f"- df_socios: {df_socios.shape}")
        print(f"- df_establecimientos: {df_establecimientos.shape}")
        print(f"- df_entrenamiento: {df_entrenamiento.shape}")

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
            
        columnas_object = df.select_dtypes(include='object').columns.tolist()
        # Exclude 'id_persona' from the list of columns to be text-cleaned
        if 'id_persona' in columnas_object:
            columnas_object.remove('id_persona') 

        # Convert remaining object columns to string and apply text cleaning
        for col in columnas_object:
            df[col] = df[col].astype(str).apply(lambda texto: re.sub(r"\s\s+", " ",
                                re.sub(r"[\r\n]+", ' ',
                                re.sub(r'\[#*.>=\]', '',
                                re.sub(r'[^a-z ]', ' ',
                                re.sub(r" \d+", ' ',
                                texto.lower()))))).strip())
        
        #df["id_persona"] = pd.to_numeric(df["id_persona"], errors="coerce")
        df['id_persona'] = pd.to_numeric(df['id_persona'], errors='coerce')
        df['id_persona'] = df['id_persona'].astype('Int64')
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

        return data_extendida_clean
    
# ------------------------------------------------------------
#                  BLOQUE PRINCIPAL
# ------------------------------------------------------------
if __name__ == "__main__":

    s3 = S3DataManager()

    BUCKET_SOURCE = 'dcelip-dev-brz-blu-s3'
    BUCKET_DEST = 'dcelip-dev-artifacts-s3'
    DESTINO = 'mlops/input/raw/'
    DESTINO_PROCESSED = 'mlops/input/processed/'

    # ------------------------------------------------------------
    #                1) GENERAR ARCHIVOS EN RAW
    # ------------------------------------------------------------

    # df_socios = s3.load_dataframe_from_s3(
    #     BUCKET_SOURCE,
    #     'source=teradata/type=socios/year=2025/month=11/day=5/'
    # )
    # if not df_socios.empty:
    #     s3.save_dataframe_to_s3(df_socios, BUCKET_DEST, DESTINO,
    #                             'df_socios_2025-11-05.parquet')

    # df_establecimientos = s3.load_dataframe_from_s3(
    #     BUCKET_SOURCE,
    #     'source=teradata/type=establecimientos/year=2025/month=11/day=5/'
    # )
    # if not df_establecimientos.empty:
    #     s3.save_dataframe_to_s3(df_establecimientos, BUCKET_DEST, DESTINO,
    #                             'df_establecimientos_2025-11-05.parquet')

    # df_demograficas = s3.load_dataframe_from_s3(
    #     BUCKET_SOURCE,
    #     'source=teradata/type=demograficas/year=2025/month=10/day=31/'
    # )
    # if not df_demograficas.empty:
    #     s3.save_dataframe_to_s3(df_demograficas, BUCKET_DEST, DESTINO,
    #                             'df_demograficas_2025-10-31.parquet')

    # df_entrenamiento = s3.load_dataframe_from_s3(
    #     BUCKET_SOURCE,
    #     'source=teradata/type=cao_entrenamiento/year=2025/month=11/day=6/'
    # )
    # if not df_entrenamiento.empty:
    #     s3.save_dataframe_to_s3(df_entrenamiento, BUCKET_DEST, DESTINO,
    #                             'df_entrenamiento_2025-11-06.parquet')
    
    # # ------------------------------------------------------------
    # #          2) IDENTIFICAR EL ARCHIVO MÁS RECIENTE POR TIPO
    # # ------------------------------------------------------------
    # logger.info("Buscando archivos más recientes en el bucket destino por tipo...")

    # archivo_socios = s3.get_newest_file_by_date(
    #     bucket_name=BUCKET_DEST,
    #     prefix=DESTINO,
    #     starts_with="df_socios"
    # )

    # archivo_establecimientos = s3.get_newest_file_by_date(
    #     bucket_name=BUCKET_DEST,
    #     prefix=DESTINO,
    #     starts_with="df_establecimientos"
    # )

    # archivo_entrenamiento = s3.get_newest_file_by_date(
    #     bucket_name=BUCKET_DEST,
    #     prefix=DESTINO,
    #     starts_with="df_entrenamiento"
    # )

    # logger.info(f"Archivo más reciente df_socios: {archivo_socios}")
    # logger.info(f"Archivo más reciente df_establecimientos: {archivo_establecimientos}")
    # logger.info(f"Archivo más reciente df_entrenamiento: {archivo_entrenamiento}")

    # # ------------------------------------------------------------
    # # Instanciar la pipeline de procesamiento
    # # ------------------------------------------------------------
    # pipeline = DataPreprocessingPipeline()

    # # ------------------------------------------------------------
    # # Cargar los archivos más recientes usando _load_raw_data
    # # ------------------------------------------------------------
    # try:
    #     df_socios, df_establecimientos, df_entrenamiento = pipeline._load_raw_data()
    # except FileNotFoundError as e:
    #     logger.error(f"Error cargando archivos: {e}")
    #     sys.exit(1)

    # # ------------------------------------------------------------
    # # Mostrar un resumen rápido de los DataFrames cargados
    # # ------------------------------------------------------------
    # logger.info(f"df_socios cargado: {df_socios.shape}")
    # logger.info(f"df_establecimientos cargado: {df_establecimientos.shape}")
    # logger.info(f"df_entrenamiento cargado: {df_entrenamiento.shape}")

    # print("\n===== Primeras filas de df_socios =====")
    # print(df_socios.head())
    # print("======================================\n")

    # print("\n===== Primeras filas de df_establecimientos =====")
    # print(df_establecimientos.head())
    # print("===============================================\n")

    # print("\n===== Primeras filas de df_entrenamiento =====")
    # print(df_entrenamiento.head())
    # print("============================================\n")

    # # ------------------------------------------------------------
    # # Instanciar la pipeline de procesamiento
    # # ------------------------------------------------------------
    pipeline = DataPreprocessingPipeline()

    # # ------------------------------------------------------------
    # # Cargar los archivos más recientes usando _load_raw_data
    # # ------------------------------------------------------------
    # try:
    #     df_socios, df_establecimientos, df_entrenamiento = pipeline._load_raw_data()
    # except FileNotFoundError as e:
    #     logger.error(f"Error cargando archivos: {e}")
    #     sys.exit(1)

    # # ------------------------------------------------------------
    # # Mostrar un resumen rápido de los DataFrames cargados
    # # ------------------------------------------------------------
    # logger.info(f"df_socios cargado: {df_socios.shape}")
    # logger.info(f"df_establecimientos cargado: {df_establecimientos.shape}")
    # logger.info(f"df_entrenamiento cargado: {df_entrenamiento.shape}")

    # print("\n===== Primeras filas de df_socios =====")
    # print(df_socios.head())
    # print("======================================\n")

    # print("\n===== Primeras filas de df_establecimientos =====")
    # print(df_establecimientos.head())
    # print("===============================================\n")

    # print("\n===== Primeras filas de df_entrenamiento =====")
    # print(df_entrenamiento.head())
    # print("============================================\n")

    # # ------------------------------------------------------------
    # # Generar el DataFrame extendido limpio usando data_extended
    # # ------------------------------------------------------------
    # logger.info("Generando DataFrame extendido limpio...")

    # data_extendida_clean = pipeline.data_extended()

    # logger.info(f"DataFrame extendido generado: { data_extendida_clean.shape}")

    # print("\n===== Primeras filas del DataFrame extendido =====")
    # print( data_extendida_clean.head())
    # print("==================================================\n")

    # # ------------------------------------------------------------
    # # Guardar el DataFrame extendido limpio en S3
    # # ------------------------------------------------------------
    # DESTINO_PROCESSED = 'mlops/input/processed/'
    # NOMBRE_ARCHIVO_EXTENDIDO = 'data_extendida_clean.parquet'

    # try:
    #     s3.save_dataframe_to_s3(
    #         df=data_extendida_clean,
    #         bucket=BUCKET_DEST,
    #         path_destino=DESTINO_PROCESSED,
    #         nombre_archivo=NOMBRE_ARCHIVO_EXTENDIDO
    #     )
    # except Exception as e:
    #     logger.error(f"No se pudo guardar el DataFrame extendido en S3: {e}")
    #     sys.exit(1)

    # ------------------------------------------------------------
    # Cargar el archivo data_extendida_clean más reciente desde S3
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    #          2) IDENTIFICAR EL ARCHIVO MÁS RECIENTE POR TIPO
    # ------------------------------------------------------------
    logger.info("Buscando archivos data_extendida_clean.py más recientes en el bucket destino por tipo...")

    df_1 = s3.get_newest_file_by_date(
        bucket_name=BUCKET_DEST,
        prefix=DESTINO_PROCESSED,
        starts_with="data_extendida_clean"
    )

    df_clean1 = s3.load_single_parquet(df_1)
    df_clean2 = pipeline.clean_recent_text_columns(df_clean1)
    df_clean3 = pipeline.outliers_filters(df_clean2)
    # df_clean4 = pipeline.normalization_establecimientos(df_clean3)
    print(df_clean3.head(4))

