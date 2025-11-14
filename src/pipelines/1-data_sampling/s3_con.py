import os
import boto3
import pandas as pd
import s3fs
from dotenv import load_dotenv

class S3DataManager:
    def __init__(self, bucket_source, bucket_dest):
        self.bucket_source = bucket_source
        self.bucket_dest = bucket_dest
        self.fs = None
        self._load_env_credentials()
        self._init_s3()

    def _load_env_credentials(self):
        """Carga las credenciales de AWS desde el archivo .env"""
        load_dotenv() 
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION')

        if not all([self.aws_access_key, self.aws_secret_key, self.aws_region]):
            raise ValueError("Faltan variables de entorno AWS en el archivo .env")

    def _init_s3(self):
        """Inicializa cliente y sistema de archivos S3"""
        boto3.client('s3')
        self.fs = s3fs.S3FileSystem(
            key=self.aws_access_key,
            secret=self.aws_secret_key
        )

    def concat_files(self, type, year, month, day):
        """Concatena todos los archivos Parquet de una carpeta S3 leyendo uno por uno"""
        prefix = f'source=teradata/type={type}/year={year}/month={month}/day={day}/'
        path_s3 = f'{self.bucket_source}/{prefix}'

        parquet_files = [f's3://{file}' for file in self.fs.ls(path_s3) if file.endswith('.parquet')]
        print(f" Archivos encontrados para '{type}': {len(parquet_files)}")

        df_list = []
        for file in parquet_files:
            df = pd.read_parquet(file, storage_options={
                'key': self.aws_access_key,
                'secret': self.aws_secret_key
            })
            df_list.append(df)

        df_concat = pd.concat(df_list, ignore_index=True)
        print(f" Registros concatenados: {df_concat.shape[0]}")

        return df_concat

    def save_bucket_data(self, path_destino, nombre_archivo, df):
        """Guarda un DataFrame como parquet en el bucket destino"""
        ruta_relativa = f"{path_destino}{nombre_archivo}"
        ruta_s3_destino = f"s3://{self.bucket_dest}/{ruta_relativa}"

        try:
            with self.fs.open(ruta_s3_destino, 'wb') as f:
                df.to_parquet(f, index=False)
            print(f"Archivo guardado correctamente en: {ruta_s3_destino}")
        except Exception as e:
            print(f"Error al guardar en S3: {e}")


# =========================
# BLOQUE PRINCIPAL DE PRUEBA
# =========================
if __name__ == "__main__":

    bucket_source = 'dcelip-dev-brz-blu-s3'
    bucket_dest = 'dcelip-dev-artifacts-s3'
    TYPES = ['socios', 'establecimientos', 'demograficas', 'cao_entrenamiento']

    ANHO_SOCIOS, MES_SOCIOS, DIA_SOCIOS = '2025', '11', '5'
    ANHO_ESTABLECIMIENTOS, MES_ESTABLECIMIENTOS, DIA_ESTABLECIMIENTOS = '2025', '11', '5'
    ANHO_DEMOGRAFICAS, MES_DEMOGRAFICAS, DIA_DEMOGRAFICAS = '2025', '10', '31'
    ANHO_ENTRENAMIENTO, MES_ENTRENAMIENTO, DIA_ENTRENAMIENTO = '2025', '11', '6'

    s3_manager = S3DataManager(bucket_source, bucket_dest)

    df_socios = s3_manager.concat_files(TYPES[0], ANHO_SOCIOS, MES_SOCIOS, DIA_SOCIOS)
    df_establecimientos = s3_manager.concat_files(TYPES[1], ANHO_ESTABLECIMIENTOS, MES_ESTABLECIMIENTOS, DIA_ESTABLECIMIENTOS)
    df_demograficas = s3_manager.concat_files(TYPES[2], ANHO_DEMOGRAFICAS, MES_DEMOGRAFICAS, DIA_DEMOGRAFICAS)
    df_entrenamiento = s3_manager.concat_files(TYPES[3], ANHO_ENTRENAMIENTO, MES_ENTRENAMIENTO, DIA_ENTRENAMIENTO)

    s3_manager.save_bucket_data('mlops/input/raw/', 'df_socios_concat.parquet', df_socios)
    s3_manager.save_bucket_data('mlops/input/raw/', 'df_establecimientos_concat.parquet', df_establecimientos)
    s3_manager.save_bucket_data('mlops/input/raw/', 'df_demograficas_concat.parquet', df_demograficas)
    s3_manager.save_bucket_data('mlops/input/raw/', 'df_entrenamiento_concat.parquet', df_entrenamiento)