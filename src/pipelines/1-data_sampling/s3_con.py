import os
import boto3
import pandas as pd
import s3fs
from datetime import datetime


class S3DataManager:
    def __init__(self, bucket_source, bucket_dest):
        self.bucket_source = bucket_source
        self.bucket_dest = bucket_dest
        self.fs = None
        self._load_env_credentials()
        self._init_s3()

    def _load_env_credentials(self):
        """Carga credenciales desde variables de entorno (GitHub Actions)."""
        self.aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.environ.get("AWS_DEFAULT_REGION")

        if not all([self.aws_access_key, self.aws_secret_key, self.aws_region]):
            raise ValueError("Faltan variables de entorno AWS.")

    def _init_s3(self):
        """Inicializa clientes S3 y filesystem."""
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.aws_region,
        )

        self.fs = s3fs.S3FileSystem(
            key=self.aws_access_key,
            secret=self.aws_secret_key,
            client_kwargs={"region_name": self.aws_region},
        )

    # ----------------------------------------------------------------------
    # Optimización: No concatenar en RAM — escribir incrementalmente
    # ----------------------------------------------------------------------
    def concat_files_streaming(self, type, year, month, day):
        """
        Lee múltiples archivos Parquet y genera un único Parquet final,
        sin cargar todo en RAM, usando escritura incremental.
        """

        prefix = f"source=teradata/type={type}/year={year}/month={month}/day={day}/"
        path_s3 = f"{self.bucket_source}/{prefix}"

        parquet_files = [
            f"s3://{file}"
            for file in self.fs.ls(path_s3)
            if file.endswith(".parquet")
        ]

        print(f"Archivos encontrados en '{type}': {len(parquet_files)}")

        if len(parquet_files) == 0:
            raise ValueError(f"No se encontraron archivos parquet para {type}.")

        # Timestamp para el nombre final
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"concat_{type}_{timestamp}.parquet"
        temp_local_file = f"/tmp/{output_file}"

        # Escritura incremental
        first = True
        for file in parquet_files:
            print(f"Leyendo archivo: {file}")
            df = pd.read_parquet(
                file,
                storage_options={
                    "key": self.aws_access_key,
                    "secret": self.aws_secret_key,
                },
            )

            # Primer archivo: crea el parquet
            if first:
                df.to_parquet(temp_local_file, index=False)
                first = False
            else:
                # Archivos siguientes: append row groups
                df.to_parquet(
                    temp_local_file,
                    index=False,
                    append=True,
                    engine="pyarrow",
                )

        print(f"Archivo concatenado generado: {temp_local_file}")
        return temp_local_file, output_file

    # ----------------------------------------------------------------------
    # SUBIR A S3
    # ----------------------------------------------------------------------
    def upload_to_s3(self, local_path, s3_path):
        """Sube un archivo local a un bucket S3."""
        try:
            self.s3_client.upload_file(local_path, self.bucket_dest, s3_path)
            print(f"Archivo subido correctamente a: s3://{self.bucket_dest}/{s3_path}")
        except Exception as e:
            print(f"Error subiendo archivo a S3: {e}")
            raise e


# =========================
# BLOQUE PRINCIPAL
# =========================
if __name__ == "__main__":

    bucket_source = "dcelip-dev-brz-blu-s3"
    bucket_dest = "dcelip-dev-artifacts-s3"
    TYPES = ["socios", "establecimientos", "demograficas", "cao_entrenamiento"]

    FECHAS = {
        "socios": ("2025", "11", "5"),
        "establecimientos": ("2025", "11", "5"),
        "demograficas": ("2025", "10", "31"),
        "cao_entrenamiento": ("2025", "11", "6"),
    }

    s3_manager = S3DataManager(bucket_source, bucket_dest)

    for t in TYPES:
        print(f"\nProcesando tipo: {t}")
        year, month, day = FECHAS[t]

        # 1) Generar parquet concatenado sin usar RAM excesiva
        local_file, filename = s3_manager.concat_files_streaming(t, year, month, day)

        # 2) Subir a S3
        destino = f"mlops/input/raw/{filename}"
        s3_manager.upload_to_s3(local_file, destino)

    print("\nPROCESO COMPLETO SIN ERRORES\n")
