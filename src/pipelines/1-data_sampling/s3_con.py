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

# ------------------------------------------------------------
#                  BLOQUE PRINCIPAL
# ------------------------------------------------------------
if __name__ == "__main__":

    s3 = S3DataManager()

    BUCKET_SOURCE = 'dcelip-dev-brz-blu-s3'
    BUCKET_DEST = 'dcelip-dev-artifacts-s3'
    DESTINO = 'mlops/input/raw/'

    # ------------------------------------------------------------
    #                1) GENERAR ARCHIVOS EN RAW
    # ------------------------------------------------------------

    df_socios = s3.load_dataframe_from_s3(
        BUCKET_SOURCE,
        'source=teradata/type=socios/year=2025/month=11/day=5/'
    )
    if not df_socios.empty:
        s3.save_dataframe_to_s3(df_socios, BUCKET_DEST, DESTINO,
                                'df_socios_2025-11-05.parquet')

    df_establecimientos = s3.load_dataframe_from_s3(
        BUCKET_SOURCE,
        'source=teradata/type=establecimientos/year=2025/month=11/day=5/'
    )
    if not df_establecimientos.empty:
        s3.save_dataframe_to_s3(df_establecimientos, BUCKET_DEST, DESTINO,
                                'df_establecimientos_2025-11-05.parquet')

    df_demograficas = s3.load_dataframe_from_s3(
        BUCKET_SOURCE,
        'source=teradata/type=demograficas/year=2025/month=10/day=31/'
    )
    if not df_demograficas.empty:
        s3.save_dataframe_to_s3(df_demograficas, BUCKET_DEST, DESTINO,
                                'df_demograficas_2025-10-31.parquet')

    df_entrenamiento = s3.load_dataframe_from_s3(
        BUCKET_SOURCE,
        'source=teradata/type=cao_entrenamiento/year=2025/month=11/day=6/'
    )
    if not df_entrenamiento.empty:
        s3.save_dataframe_to_s3(df_entrenamiento, BUCKET_DEST, DESTINO,
                                'df_entrenamiento_2025-11-06.parquet')
    
    # ------------------------------------------------------------
    #          2) IDENTIFICAR EL ARCHIVO MÁS RECIENTE POR TIPO
    # ------------------------------------------------------------

    logger.info("Buscando archivos más recientes en el bucket destino por tipo...")

    archivo_socios = s3.get_newest_file_by_date(
        bucket_name=BUCKET_DEST,
        prefix=DESTINO,
        starts_with="df_socios"
    )

    archivo_establecimientos = s3.get_newest_file_by_date(
        bucket_name=BUCKET_DEST,
        prefix=DESTINO,
        starts_with="df_establecimientos"
    )

    archivo_entrenamiento = s3.get_newest_file_by_date(
        bucket_name=BUCKET_DEST,
        prefix=DESTINO,
        starts_with="df_entrenamiento"
    )

    logger.info(f"Archivo más reciente df_socios: {archivo_socios}")
    logger.info(f"Archivo más reciente df_establecimientos: {archivo_establecimientos}")
    logger.info(f"Archivo más reciente df_entrenamiento: {archivo_entrenamiento}")