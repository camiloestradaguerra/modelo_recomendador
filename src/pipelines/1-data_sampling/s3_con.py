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
    """Pipeline de preprocesamiento de datos para el sistema de recomendación.

    Buenas prácticas aplicadas:
    - Uso de `logger` en lugar de `print`.
    - Validación temprana de variables de entorno.
    - Parámetros configurables (bucket y rutas) con valores por defecto.
    - Docstrings en métodos y manejo de excepciones para pasos críticos.
    """

    def __init__(self,
                 bucket_name: str = 'dcelip-dev-artifacts-s3',
                 file_raw_path: str = 'mlops/input/raw/',
                 file_procesed_path: str = 'mlops/input/processed/') -> None:
        """Inicializa la pipeline.

        Parametros:
        - bucket_name: nombre del bucket destino por defecto.
        - file_raw_path: prefijo donde están los archivos raw.
        - file_procesed_path: prefijo para archivos procesados.
        """

        self.aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.environ.get("AWS_DEFAULT_REGION")

        if not all([self.aws_access_key, self.aws_secret_key, self.aws_region]):
            logger.error("Faltan variables de entorno AWS en el archivo .env o variables de entorno del sistema.")
            raise ValueError("Faltan variables de entorno AWS en el archivo .env")

        self.bucket_name = bucket_name
        self.file_raw_path = file_raw_path
        self.file_procesed_path = file_procesed_path

        logger.info("DataPreprocessingPipeline inicializada correctamente. bucket=%s prefix_raw=%s",
                    self.bucket_name, self.file_raw_path)
        
    def _load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw data from the newest parquet files in S3 using S3DataManager."""
        logger.info("Configurando acceso a S3 para cargar datos raw...")

        try:
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
                logger.error("No se encontraron archivos recientes para todos los datasets.")
                raise FileNotFoundError("No se encontraron archivos recientes para todos los datasets.")

            logger.info("Leyendo archivos más recientes desde S3...")

            # Validar que las URIs existen antes de leer (get_newest_file_by_date ya retorna None si no hay)
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

            logger.info("Archivos cargados correctamente: - df_socios: %s - df_establecimientos: %s - df_entrenamiento: %s",
                        df_socios.shape, df_establecimientos.shape, df_entrenamiento.shape)

            self.df_socios = df_socios
            self.df_establecimientos = df_establecimientos
            self.df_entrenamiento = df_entrenamiento

            return df_socios, df_establecimientos, df_entrenamiento

        except Exception as e:
            logger.exception("Error cargando datos raw desde S3: %s", e)
            raise
        finally:
            # aquí podríamos limpiar recursos si fuera necesario
            pass
    
    def data_extended(self) -> pd.DataFrame:
        """Cleans and merges socios, establecimientos, and entrenamiento datasets"""
        logger.info("Iniciando proceso de limpieza y merge de datasets...")

        try:
            df_socios = self.df_socios
            df_establecimientos = self.df_establecimientos
            df_entrenamiento = self.df_entrenamiento

            # Validaciones básicas de tipos
            if not isinstance(df_socios, pd.DataFrame) or not isinstance(df_establecimientos, pd.DataFrame) or not isinstance(df_entrenamiento, pd.DataFrame):
                logger.error("Los objetos de entrada deben ser DataFrame. Revise _load_raw_data o las asignaciones previas.")
                raise TypeError("Entradas a data_extended deben ser pandas.DataFrame")

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
            df_entrenamiento['ID_ESTABLECIMIENTO'] = pd.to_numeric(df_entrenamiento['ID_ESTABLECIMIENTO'], errors='coerce')
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
            data_extendida = (
                df_merge1.merge(df_establecimientos, on='ID_ESTABLECIMIENTO', how='left')
                .dropna(subset=['CADENA', 'ESTABLECIMIENTO', 'ESPECIALIDAD', 'LOCALIZACION_EXTERNA'])
            )

            logger.info("Data extendida generada con shape: %s", data_extendida.shape)
            return data_extendida

        except Exception as e:
            logger.exception("Error durante data_extended: %s", e)
            raise
        finally:
            # Se podría medir duración o liberar memoria intermedia
            pass

    def clean_recent_text_columns(self, df) -> pd.DataFrame:
        """Normaliza y limpia columnas de texto del DataFrame.

        - Pasa a minúsculas, elimina saltos de línea, caracteres no alfabéticos y números sueltos.
        - Convierte columnas específicas a tipos apropiados.
        """

        logger.info("Limpiando columnas de texto recientes. Columnas iniciales: %s", df.shape[1])

        try:
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

            logger.info("Limpieza de texto completada.")
            return df

        except Exception as e:
            logger.exception("Error en clean_recent_text_columns: %s", e)
            raise
    
    def outliers_filters(self, df) -> pd.DataFrame:
        """Aplica filtros heurísticos para reducir outliers y ciudades poco representadas.

        Retorna un DataFrame filtrado.
        """

        logger.info("Aplicando filtros de outliers...")

        try:
            df = df[df['antiguedad_socio_unico']<df['antiguedad_socio_unico'].quantile(0.99)].reset_index(drop=True)

            df_by_city = df.groupby('ciudad')['antiguedad_socio_unico'].size().reset_index().sort_values(by='antiguedad_socio_unico', ascending=False)

            ciudades_con_alta_antiguedad_df = df_by_city[df_by_city['antiguedad_socio_unico'] > 25]
            nombres_de_ciudades = ciudades_con_alta_antiguedad_df['ciudad'].unique()

            df_filtrado = df[df['ciudad'].isin(nombres_de_ciudades)]

            df_frec_cadena = df_filtrado['cadena'].value_counts(ascending=False).reset_index()

            filtro_cadena = df_frec_cadena[df_frec_cadena['count'] < 7]['cadena'].unique()
            
            df_filtrado_cadena = df_filtrado[~df_filtrado['cadena'].isin(filtro_cadena)]

            logger.info("Outliers filtrados. Resultado shape: %s", df_filtrado_cadena.shape)
            return df_filtrado_cadena

        except Exception as e:
            logger.exception("Error en outliers_filters: %s", e)
            raise

    def normalization_establecimientos(self, df) -> pd.DataFrame:

        if not all([self.aws_access_key, self.aws_secret_key, self.aws_region]):
            logger.error("Faltan variables de entorno AWS. Verifica que AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY y AWS_DEFAULT_REGION estén definidos.")
            raise EnvironmentError(
                "Faltan variables de entorno AWS. Verifica que AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY y AWS_DEFAULT_REGION estén definidos."
        )

        logger.info("Normalizando nombres de establecimientos...")

        try:
            s3 = boto3.client('s3')
            fs = s3fs.S3FileSystem()
            
            diccionario_establecimientos = {}

            dict_path = Path("diccionario_establecimientos.txt")
            if not dict_path.exists():
                logger.error("No se encontró 'diccionario_establecimientos.txt' en el directorio actual: %s", dict_path.resolve())
                raise FileNotFoundError(f"diccionario_establecimientos.txt no existe: {dict_path}")

            with dict_path.open("r", encoding="utf-8") as f:
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

            logger.info("Normalización de establecimientos completada. Resultado shape: %s", data_extendida_clean.shape)
            return data_extendida_clean

        except Exception as e:
            logger.exception("Error en normalization_establecimientos: %s", e)
            raise


# ============================================================
#                   CLI ENTRY POINT
# ============================================================

def run_preprocessing(input_path: str, output_path: str) -> None:
    """
    Execute the data preprocessing pipeline.

    This is the main entry point that orchestrates loading raw data,
    applying preprocessing transformations (text cleaning, outlier filtering,
    establishment normalization), and saving the cleaned result. It's designed
    to be called by MLflow or other orchestration tools, and supports both
    local file paths and S3 URIs.

    Parameters
    ----------
    input_path : str
        Path to the raw input data (parquet format). Can be a local path
        or an S3 URI starting with 's3://'.
    output_path : str
        Path where preprocessed data will be saved (parquet format). Can be
        a local path or an S3 URI starting with 's3://'.

    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist (for local paths).
    EnvironmentError
        If required AWS environment variables are missing (when using S3).
    Exception
        If any step in the pipeline fails during preprocessing.

    Examples
    --------
    >>> run_preprocessing(
    ...     input_path='data/raw/data_extendida.parquet',
    ...     output_path='data/processed/data_extendida_clean.parquet'
    ... )

    >>> run_preprocessing(
    ...     input_path='s3://my-bucket/raw/data.parquet',
    ...     output_path='s3://my-bucket/processed/data_clean.parquet'
    ... )

    Notes
    -----
    - The function automatically applies three preprocessing steps in order:
      1. Text column cleaning and normalization
      2. Outlier filtering based on heuristics
      3. Establishment name normalization using a dictionary
    - Results are logged to both stderr and logs/pipeline.log.
    - Input validation occurs automatically for local paths.
    - S3 operations require AWS credentials (AWS_ACCESS_KEY_ID,
      AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION) in .env or environment.
    """
    logger.info("Starting preprocessing pipeline. Loading data from input_path=%s", input_path)
    
    # Validar credenciales AWS solo si se necesitan (input o output S3)
    needs_s3 = str(input_path).startswith('s3://') or str(output_path).startswith('s3://')
    s3_manager = None

    if needs_s3:
        logger.info("S3 paths detected. Initializing S3DataManager...")
        try:
            s3_manager = S3DataManager()
        except ValueError as e:
            logger.error("Error al cargar credenciales AWS: %s", str(e))
            raise EnvironmentError("No se pudieron cargar las credenciales AWS desde .env o sistema.") from e

    logger.info("AWS credentials and S3 access validated successfully")

    # # Read input data from s3
    # logger.info("Step 1: Loading raw data from %s", input_path)

    # try:
    #     pipeline = DataPreprocessingPipeline()
    #     pipeline._load_raw_data()
    #     logger.info("Raw data concated successfully.")
    # except Exception as e:
    #     logger.exception("Failed to load raw data: %s", e)
    #     raise

    # # Build extended data

    # logger.info("Step 2: Generating extended dataset")

    # try:
    #     df_extended = pipeline.data_extended()
    #     logger.info("Extended data generated successfully. Records: %d | Columns: %d", len(df_extended), len(df_extended.columns))
    # except Exception as e:
    #     logger.exception("Failed to generate extended dataset: %s", e)
    #     raise

    # Read input from local or S3
    if str(input_path).startswith('s3://'):
        try:
            s3_reader = S3DataManager()
            df = s3_reader.load_single_parquet(input_path)
            logger.info("Data loaded from S3")
        except Exception as e:
            logger.exception("Failed to load data from S3: %s", e)
            raise
    else:
        p = Path(input_path)
        if not p.exists():
            logger.error("Input file not found at: %s", p.resolve())
            raise FileNotFoundError(f"Input file not found: {p}")
        try:
            df = pd.read_parquet(p)
            logger.info("Data loaded from local path")
        except Exception as e:
            logger.exception("Failed to read local parquet file: %s", e)
            raise

    logger.info("Loaded %d records, %d columns", len(df), len(df.columns))

    # Apply preprocessing pipeline steps
    try:
        logger.info("Step 1/3: Cleaning text columns...")
        df_clean1 = pipeline.clean_recent_text_columns(df_extended)
        
        logger.info("Step 2/3: Filtering outliers...")
        df_clean2 = pipeline.outliers_filters(df_clean1)
        
        logger.info("Step 3/3: Normalizing establishments...")
        df_clean3 = pipeline.normalization_establecimientos(df_clean2)
        
    except Exception as e:
        logger.exception("Error during preprocessing steps: %s", e)
        raise

    # Save output to local or S3
    try:
        if str(output_path).startswith('s3://'):
            uri = output_path.replace('s3://', '')
            parts = uri.split('/')
            bucket = parts[0]
            path_destino = '/'.join(parts[1:-1]) + ('/' if len(parts) > 2 else '')
            nombre_archivo = parts[-1]

            s3_writer = S3DataManager()
            s3_writer.save_dataframe_to_s3(
                df=df_clean3,
                bucket=bucket,
                path_destino=path_destino,
                nombre_archivo=nombre_archivo
            )
            logger.success("Cleaned data saved to S3 at %s", output_path)
        else:
            outp = Path(output_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            df_clean3.to_parquet(outp, index=False)
            logger.success("Cleaned data saved to local path: %s", outp.resolve())
            
    except Exception as e:
        logger.exception("Failed to save output: %s", e)
        raise

    logger.info("Preprocessing completed. Output shape: %s", df_clean3.shape)


def main():
    """
    Parse command-line arguments and run the preprocessing pipeline.

    This function serves as the entry point when the module is run as a script.
    It defines the CLI interface and delegates to the main preprocessing function.
    All arguments are required and support both local and S3 paths.

    Command-line Arguments
    ----------------------
    --input_path : str, required
        Path to the raw input parquet file (local or s3://).
    --output_path : str, required
        Path where preprocessed data will be saved (local or s3://).

    Examples
    --------
    $ python s3_con.py \\
        --input_path data/raw/data_extendida.parquet \\
        --output_path data/processed/data_extendida_clean.parquet

    $ python s3_con.py \\
        --input_path s3://bucket/raw/data.parquet \\
        --output_path s3://bucket/processed/data_clean.parquet

    Environment Variables
    ----------------------
    AWS_ACCESS_KEY_ID : str
        AWS access key (required if using S3 URIs).
    AWS_SECRET_ACCESS_KEY : str
        AWS secret key (required if using S3 URIs).
    AWS_DEFAULT_REGION : str
        AWS region (required if using S3 URIs).

    Exit Codes
    ----------
    0 : Success; preprocessing completed without errors.
    1 : Failure; an error occurred during execution.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data preprocessing pipeline for recommendation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local input and output
  python s3_con.py --input_path data/raw/sample.parquet --output_path data/processed/sample_clean.parquet

  # S3 input and output
  python s3_con.py --input_path s3://bucket/raw/data.parquet --output_path s3://bucket/processed/data_clean.parquet

  # Mixed local input with S3 output
  python s3_con.py --input_path data/raw/sample.parquet --output_path s3://bucket/processed/sample_clean.parquet
        """
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to raw input data (local or s3://)"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where preprocessed data will be saved (local or s3://)"
    )

    args = parser.parse_args()

    try:
        logger.info("Starting preprocessing pipeline with input=%s, output=%s", 
                   args.input_path, args.output_path)
        run_preprocessing(
            input_path=args.input_path,
            output_path=args.output_path
        )
        logger.success("Preprocessing pipeline completed successfully")
    except Exception as e:
        logger.exception("Preprocessing pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()