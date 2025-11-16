"""
Pipeline de preprocesamiento de datos para sistema de recomendación.

Este módulo proporciona herramientas para:
- Gestión de operaciones S3 (lectura/escritura de DataFrames)
- Preprocesamiento de datos (limpieza, filtrado, normalización)
- Orquestación del pipeline mediante CLI
"""
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
from typing import Optional, Tuple

# ============================================================
#                  CONSTANTES Y CONFIGURACIÓN
# ============================================================
LOG_FILE_PATH = Path("logs/pipeline.log")
LOG_ROTATION = "1 day"
LOG_RETENTION = "7 days"
DICT_ESTABLISHMENTS_FILE = "diccionario_establecimientos.txt"

# ============================================================
#                  CONFIGURACIÓN DEL LOGGER
# ============================================================
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
    level="INFO"
)
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_FILE_PATH,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="INFO",
    enqueue=True
)

load_dotenv()

# ------------------------------------------------------------
#                    S3 DATA MANAGER
# ------------------------------------------------------------
class S3DataManager:
    """Gestor de operaciones S3 para lectura/escritura de DataFrames.
    
    Maneja autenticación AWS, inicialización de conexiones S3,
    y operaciones de carga/guardado de archivos parquet.
    """
    
    def __init__(self) -> None:
        """Inicializa el gestor S3 cargando credenciales y conectando a AWS."""
        self.fs: Optional[s3fs.S3FileSystem] = None
        self._load_env_credentials()
        self._init_s3()

    def _load_env_credentials(self) -> None:
        """Carga credenciales AWS desde .env o variables del sistema.
        
        Raises
        ------
        ValueError
            Si faltan variables de entorno AWS requeridas.
        """
        load_dotenv()
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION')

        if not all([self.aws_access_key, self.aws_secret_key, self.aws_region]):
            missing_vars = []
            if not self.aws_access_key:
                missing_vars.append('AWS_ACCESS_KEY_ID')
            if not self.aws_secret_key:
                missing_vars.append('AWS_SECRET_ACCESS_KEY')
            if not self.aws_region:
                missing_vars.append('AWS_DEFAULT_REGION')
            raise ValueError(f"Faltan variables de entorno AWS: {', '.join(missing_vars)}")

        logger.info("Credenciales AWS cargadas correctamente.")

    def _init_s3(self) -> None:
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

    def load_dataframe_from_s3(self, bucket: str, prefix: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Carga y concatena archivos Parquet desde S3 con un prefijo.
        
        Parameters
        ----------
        bucket : str
            Nombre del bucket S3
        prefix : str
            Prefijo de la ruta en S3
        limit : Optional[int]
            Número máximo de archivos a cargar (default: None = cargar todos)
            
        Returns
        -------
        pd.DataFrame
            DataFrame concatenado con todos los parquets encontrados
        """
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
        for i, file in enumerate(parquet_files[:24]):
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
        """Carga un único archivo parquet desde un S3 URI exacto.
        
        Parameters
        ----------
        s3_uri : str
            URI completo del archivo S3 (ej: 's3://bucket/path/file.parquet')
            
        Returns
        -------
        pd.DataFrame
            DataFrame con contenido del archivo parquet
        """
        df = pd.read_parquet(
            s3_uri,
            storage_options={
                "key": self.aws_access_key,
                "secret": self.aws_secret_key
            }
        )
        return df

    def save_dataframe_to_s3(self, df: pd.DataFrame, bucket: str, path_destino: str, nombre_archivo: str) -> None:
        """Guarda un DataFrame en S3 con timestamp automático.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a guardar
        bucket : str
            Nombre del bucket S3
        path_destino : str
            Ruta destino en S3 (ej: 'mlops/input/processed/')
        nombre_archivo : str
            Nombre del archivo con extensión (ej: 'data.parquet')
            
        Raises
        ------
        Exception
            Si hay error durante la escritura en S3
        """
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
    
    def get_newest_file_by_date(self, bucket_name: str, prefix: str = "", starts_with: str = "") -> str:
        """Retorna la ruta del archivo más reciente dentro de un prefijo S3.
        
        Parameters
        ----------
        bucket_name : str
            Nombre del bucket S3
        prefix : str
            Prefijo de búsqueda en S3 (default: '')
        starts_with : str
            Filtrar por archivos que comiencen con este string (default: '')
            
        Returns
        -------
        str
            Ruta S3 del archivo más reciente (formato: 's3://bucket/key')
            
        Notes
        -----
        Busca fechas en formato YYYY-MM-DD, YYYYMMDD o YYYYMMDD_HHMMSS en el nombre.
        Si no encuentra fecha en el nombre, usa LastModified como fallback.
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
    - Inicialización sin dependencias externas (lazy loading de credenciales).
    - Docstrings en métodos y manejo robusto de excepciones.
    - Type hints para parámetros y retornos.
    """

    def __init__(self) -> None:
        """Inicializa la pipeline sin validar credenciales AWS.
        
        Las credenciales se validan solo cuando se necesiten (en métodos que accedan a S3).
        Esto permite usar la pipeline con datos locales sin requerir credenciales AWS.
        """
        logger.info("DataPreprocessingPipeline inicializada.")

    def data_extended(self, df_socios: pd.DataFrame, df_establecimientos: pd.DataFrame, df_entrenamiento: pd.DataFrame) -> pd.DataFrame:
        """Cleans and merges socios, establecimientos, and entrenamiento datasets"""
        logger.info("Iniciando proceso de limpieza y merge de datasets...")

        try:
            
            # --- Limpieza socios ---
            df_socios = df_socios[['Id_Persona', 'ESTADO_CIVIL', 'Edad', 'GENERO', 'ROL',
                                'Antiguedad_Socio_Unico', 'SEGMENTO_COMERCIAL', 'Ciudad',
                                'Zona', 'Region']].copy()
            df_socios['Id_Persona'] = pd.to_numeric(df_socios['Id_Persona'], errors='coerce')
            df_socios['Antiguedad_Socio_Unico'] = df_socios['Antiguedad_Socio_Unico'].fillna(
                df_socios['Antiguedad_Socio_Unico'].mean()
            )

            for col in ['Ciudad', 'Zona', 'Region']:
                moda = df_socios[col].mode(dropna=True)[0]
                df_socios[col] = df_socios[col].fillna(moda)

            # --- Limpieza establecimientos ---
            df_establecimientos = df_establecimientos[['ID_ESTABLECIMIENTO', 'CADENA', 'ESTABLECIMIENTO']].copy()
            df_establecimientos['ID_ESTABLECIMIENTO'] = pd.to_numeric(df_establecimientos['ID_ESTABLECIMIENTO'], errors="coerce")

            # --- Limpieza entrenamiento ---
            df_entrenamiento = df_entrenamiento[['DiaID', 'Id_Persona', 'ID_ESTABLECIMIENTO',
                                                'ESPECIALIDAD', 'HORA_INICIO', 'HORA_FIN',
                                                'LOCALIZACION_EXTERNA', 'MONTO', 'Neteo_Mensual', 'Neteo_Diario']].copy()
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

            logger.info("Data extendida generada con shape: {}".format(data_extendida.shape))
            return data_extendida

        except Exception as e:
            logger.exception("Error durante data_extended: %s", e)
            raise
        finally:
            # Se podría medir duración o liberar memoria intermedia
            pass

    def clean_recent_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza y limpia columnas de texto del DataFrame.

        - Pasa a minúsculas, elimina saltos de línea, caracteres no alfabéticos y números sueltos.
        - Convierte columnas específicas a tipos apropiados.
        """

        logger.info("Limpiando columnas de texto recientes. Columnas iniciales: {}".format(df.shape[1]))

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
    
    def outliers_filters(self, df: pd.DataFrame) -> pd.DataFrame:
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

            logger.info("Outliers filtrados. Resultado shape: {}".format(df_filtrado_cadena.shape))
            return df_filtrado_cadena

        except Exception as e:
            logger.exception("Error en outliers_filters: %s", e)
            raise

    def normalization_establecimientos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres de establecimientos usando un diccionario local.
        
        Operaciones:
        1. Carga diccionario desde 'diccionario_establecimientos.txt'
        2. Mapea nombres de establecimientos usando el diccionario
        3. Filtra establecimientos poco representados (count < 7)
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columna 'establecimiento' a normalizar.
            
        Returns
        -------
        pd.DataFrame
            DataFrame con establecimientos normalizados y filtrados.
            
        Raises
        ------
        FileNotFoundError
            Si no existe 'diccionario_establecimientos.txt'.
        Exception
            Si ocurre un error durante la normalización.
        """
        logger.info("Normalizando nombres de establecimientos...")

        try:
            diccionario_establecimientos = {}

            dict_path = Path("diccionario_establecimientos.txt")
            if not dict_path.exists():
                logger.error("No se encontró 'diccionario_establecimientos.txt' en: %s", dict_path.resolve())
                raise FileNotFoundError(f"diccionario_establecimientos.txt no existe en: {dict_path.resolve()}")

            # Cargar diccionario desde archivo
            with dict_path.open("r", encoding="utf-8") as f:
                for linea in f:
                    if ":" in linea:
                        clave, valor = linea.strip().split(":", 1)
                        diccionario_establecimientos[clave.strip()] = valor.strip()
            
            logger.info("Diccionario cargado: {} entradas".format(len(diccionario_establecimientos)))

            # Normalizar establecimientos
            df['establecimiento'] = df['establecimiento'].astype(str).str.strip().str.lower()
            df['establecimiento'] = df['establecimiento'].map(diccionario_establecimientos).fillna(df['establecimiento'])

            # Filtrar establecimientos poco representados
            df_establecimiento = df['establecimiento'].value_counts(ascending=False).reset_index()
            filtro_establecimiento = df_establecimiento[df_establecimiento['count'] < 7]['establecimiento'].unique()
            df_filtrado = df[~df['establecimiento'].isin(filtro_establecimiento)]

            logger.info("Normalización completada. Shape: {} -> {}".format(df.shape, df_filtrado.shape))
            return df_filtrado

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.exception("Error en normalization_establecimientos: %s", e)
            raise


# ============================================================
#                   CLI ENTRY POINT
# ============================================================

def _validate_aws_credentials() -> None:
    """Valida que las credenciales AWS estén disponibles en el entorno.
    
    Raises
    ------
    EnvironmentError
        Si faltan AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY o AWS_DEFAULT_REGION.
    """
    load_dotenv()
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error("Faltan variables de entorno AWS: %s", ', '.join(missing_vars))
        raise EnvironmentError(f"Credenciales AWS incompletas. Falta: {', '.join(missing_vars)}")
    
    logger.info("Credenciales AWS validadas correctamente.")


def _load_raw_data_from_s3() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga datos raw desde S3 y los concatena.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_socios, df_establecimientos, df_entrenamiento) concatenados.
        
    Raises
    ------
    Exception
        Si ocurre un error al cargar o guardar datos en S3.
    """
    logger.info("Cargando datos raw desde S3...")
    
    s3 = S3DataManager()

    # Allow CI / runtime configuration via environment variables so
    # GitHub Actions (or any CI) can inject buckets/paths without code edits.
    bucket_input = os.getenv('BUCKET_INPUT', 'dcelip-dev-brz-blu-s3')
    bucket_output = os.getenv('BUCKET_OUTPUT', 'dcelip-dev-artifacts-s3')
    path_raw = os.getenv('PATH_RAW', 'mlops/input/raw/')
    
    # Definir prefijos de entrada
    prefixes = {
        'socios': 'source=teradata/type=socios/year=2025/month=11/day=5/',
        'establecimientos': 'source=teradata/type=establecimientos/year=2025/month=11/day=5/',
        'entrenamiento': 'source=teradata/type=cao_entrenamiento/year=2025/month=11/day=7/'
    }
    
    # Cargar y concatenar cada tipo de dato
    df_socios = s3.load_dataframe_from_s3(bucket=bucket_input, prefix=prefixes['socios'])
    df_establecimientos = s3.load_dataframe_from_s3(bucket=bucket_input, prefix=prefixes['establecimientos'])
    df_entrenamiento = s3.load_dataframe_from_s3(bucket=bucket_input, prefix=prefixes['entrenamiento'])
    
    logger.info("Datos cargados - socios: {} | establecimientos: {} | entrenamiento: {}".format(
        df_socios.shape, df_establecimientos.shape, df_entrenamiento.shape))
    
    # Guardar datos concatenados en bucket de artifacts
    logger.info("Guardando datos concatenados en S3...")
    s3.save_dataframe_to_s3(df_socios, bucket_output, path_raw, 'df_socios.parquet')
    s3.save_dataframe_to_s3(df_establecimientos, bucket_output, path_raw, 'df_establecimientos.parquet')
    s3.save_dataframe_to_s3(df_entrenamiento, bucket_output, path_raw, 'df_entrenamiento.parquet')
    
    logger.info("Datos concatenados guardados correctamente.")
    return df_socios, df_establecimientos, df_entrenamiento


def run_preprocessing(output_path: str) -> None:
    """
    Execute the data preprocessing pipeline.

    This is the main entry point that orchestrates:
    1. Loading raw data from S3
    2. Building extended dataset
    3. Applying preprocessing transformations (text cleaning, outlier filtering, normalization)
    4. Saving cleaned result to local or S3

    Parameters
    ----------
    output_path : str
        Path where preprocessed data will be saved (parquet format).
        Can be a local path or an S3 URI starting with 's3://'.

    Raises
    ------
    FileNotFoundError
        If required files don't exist.
    EnvironmentError
        If AWS credentials are missing when needed.
    Exception
        If any step in the pipeline fails.

    Examples
    --------
    >>> run_preprocessing(
    ...     output_path='data/processed/data_extendida_clean.parquet'
    ... )

    >>> run_preprocessing(
    ...     output_path='s3://bucket/processed/data_clean.parquet'
    ... )

    Notes
    -----
    - The function automatically applies preprocessing in order:
      1. Load raw data from S3 and build extended dataset
      2. Clean text columns
      3. Filter outliers
      4. Normalize establishment names
    - All operations are logged to stderr and logs/pipeline.log.
    - S3 operations require AWS credentials in .env or environment variables.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE PREPROCESAMIENTO")
    logger.info("=" * 60)
    
    try:
        # Validar credenciales AWS (siempre se necesitan para cargar datos raw)
        logger.info("Validando credenciales AWS...")
        _validate_aws_credentials()
        
        # Paso 1: Cargar datos raw desde S3
        logger.info("\n[PASO 1/4] Cargando datos raw desde S3...")
        df_socios, df_establecimientos, df_entrenamiento = _load_raw_data_from_s3()
        
        # Paso 2: Construir dataset extendido
        logger.info("\n[PASO 2/4] Construyendo dataset extendido...")
        pipeline = DataPreprocessingPipeline()
        df_extendida = pipeline.data_extended(
            df_socios=df_socios,
            df_establecimientos=df_establecimientos,
            df_entrenamiento=df_entrenamiento
        )
        logger.info("Dataset extendido creado: {}".format(df_extendida.shape))
        
        # Paso 3: Aplicar transformaciones de preprocesamiento
        logger.info("\n[PASO 3/4] Aplicando transformaciones de preprocesamiento...")
        
        df_clean1 = pipeline.clean_recent_text_columns(df_extendida)
        logger.info("  ✓ Limpieza de texto completada: {}".format(df_clean1.shape))
        
        df_clean2 = pipeline.outliers_filters(df_clean1)
        logger.info("  ✓ Filtrado de outliers completado: {}".format(df_clean2.shape))
        
        df_clean3 = pipeline.normalization_establecimientos(df_clean2)
        logger.info("  ✓ Normalización de establecimientos completada: {}".format(df_clean3.shape))
        
        # Paso 4: Guardar resultado
        logger.info("\n[PASO 4/4] Guardando resultado...")
        
        if str(output_path).startswith('s3://'):
            # Parsear S3 URI
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
            logger.success("Datos limpios guardados en S3: {}".format(output_path))
        else:
            # Guardar localmente
            outp = Path(output_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            df_clean3.to_parquet(outp, index=False)
            logger.success("Datos limpios guardados localmente: {}".format(outp.resolve()))
        
        logger.info("\n" + "=" * 60)
        logger.success("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("Registros procesados: {} | Columnas: {}".format(len(df_clean3), len(df_clean3.columns)))
        logger.info("=" * 60)
        
    except EnvironmentError as e:
        logger.error("Error de configuración: %s", str(e))
        raise
    except FileNotFoundError as e:
        logger.error("Archivo no encontrado: %s", str(e))
        raise
    except Exception as e:
        logger.exception("Error inesperado durante el pipeline: %s", e)
        raise


def main():
    """
    Parse command-line arguments and run the preprocessing pipeline.

    This function serves as the entry point when the module is run as a script.
    It defines the CLI interface with proper help text and error handling.

    Command-line Arguments
    ----------------------
    --output_path : str, required
        Path where preprocessed data will be saved (local or s3://).

    Examples
    --------
    $ python s3_con.py --output_path data/processed/data_clean.parquet

    $ python s3_con.py --output_path s3://bucket/processed/data_clean.parquet

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
  # Save to local path
  python s3_con.py --output_path data/processed/sample_clean.parquet

  # Save to S3
  python s3_con.py --output_path s3://bucket/processed/sample_clean.parquet

Requirements:
  - AWS credentials in .env or environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)
  - diccionario_establecimientos.txt in the current directory
        """
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where preprocessed data will be saved (local or s3://)"
    )

    args = parser.parse_args()

    try:
        logger.info("CLI invocada con output_path=%s", args.output_path)
        run_preprocessing(output_path=args.output_path)
        logger.success("Pipeline completado exitosamente")
    except EnvironmentError as e:
        logger.error("Error de configuración: %s", str(e))
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error("Archivo requerido no encontrado: %s", str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception("Error inesperado: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()