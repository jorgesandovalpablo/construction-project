from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

SCHEMAS =  {
    "historico_equipos":{
        "columns":[
            "Date","Price_X","Price_Y",
            "Price_Z","Price_Equipo1","Price_Equipo2"
        ],
        "dtypes":{
            "Price_X":    "float64",
            "Price_Y":    "float64",
            "Price_Z":    "float64",
            "Price_Equipo1":  "float64",
            "Price_Equipo2":  "float64"
        },
        "parse_dates": ["Date"],
    },
    "X":{
        "columns":[
            "Date","Price_X"
        ],
        "dtypes":{
            "Price_X":      "float64",
        },
        "parse_dates": ["Date"],
        "rename": {"Price": "Price_X"},
    },
    "Y":{
        "columns":[
            "Date","Price_Y"
        ],
        "dtypes":{
            "Price_Y":      "float64",
        },
        "parse_dates": ["Date"],
        "sep": ";",
        "dayfirst": True,
        "decimal": ",",
        "rename": {"Price": "Price_Y"},
    },
    "Z":{
        "columns":[
            "Date","Price_Z"
        ],
        "dtypes":{
            "Price_Z":      "float64",
        },
        "parse_dates": ["Date"],
        "rename": {"Price": "Price_Z"},
    }
}

def _validate_columns(df:pd.DataFrame, name:str, expected_cols: list) -> None:
    """
    Valida que el dataframe cumpla respecto al esquema
    """
    missing = set(expected_cols) -  set(df.columns)
    extra =  set(df.columns) - set (expected_cols)

    if missing:
        logger.error(f"{name}: columnas faltantes : {missing}")
        raise ValueError(
            f"Columnas faltantes en {name}: {missing}"
        )
    if extra:
        logger.warning(f"{name}: columnas extras detectadas : {extra}")

def _validate_nulls(df:pd.DataFrame, name:str) -> None:
    """Reporta nulos por columna"""
    nulls =  df.isnull().sum()
    nulls = nulls[nulls > 0]

    if not nulls.empty:
        for col, count in nulls.items():
            pct = count / len(df) * 100
            logger.warning(
                f"{name}: nulos ene '{col}' -> "
                f"{count:,}({pct:.1f}%)"
            )
    else:
        logger.info(f"{name}: sin nulos")

def _validate_schema(df:pd.DataFrame, name: str, schema: dict) -> None:
    """
    Ejecuta validaciones sobre el DF
    """
    logger.info(f"Validando esuqema de {name}...")
    _validate_columns(df,name,schema["columns"])
    _validate_nulls(df,name)
    logger.info(
        f"{name}: esuqema valido |"
        f"shape: {df.shape}"
    )

# -------------------
# Ingesta
# -------------------

def _load_csv(name:str, data_path: Path) -> pd.DataFrame:
    """
    Carga Un CSV, valida el esquema y retorna un DataFrame

    Params:
        name:       nombre del archivo sin extension
        data_path:  ruta base del archivo csv

    Retorna:
        Dataframe valido
    """

    path= data_path / f"{name}.csv"
    schema = SCHEMAS[name]

    if not path.exists():
        logger.error(f"Archivo no encotrado en:{path}")
        raise FileNotFoundError(f"No se encontro: {path}")

    logger.info(f"Cargando {name}.csv")
    df = pd.read_csv(
        path,
        sep=schema.get("sep", ","),
        decimal=schema.get("decimal", "."),
    )
    if "rename" in schema:
        df = df.rename(columns=schema["rename"])
    for date_col in schema["parse_dates"]:
        df[date_col] = pd.to_datetime(
            df[date_col],
            dayfirst=schema.get("dayfirst", False),
            errors="coerce",
        )
    for col, dtype in schema["dtypes"].items():
        if col not in df.columns:
            continue
        if dtype.startswith("float"):
            before = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            coerced = df[col].isna().sum() - before
            if coerced:
                logger.warning(
                    f"{name}.{col}: {coerced} valores no numéricos "
                    f"convertidos a NaN"
                )
        else:
            df[col] = df[col].astype(dtype)
    _validate_schema(df, name, schema)
    return df


def load_raw_data(data_path:str = None) -> dict:
    """
    Carga los archivos csv históricos.
    params: data_path: ruta a la carpeta de csv's
    si es none, se usan los valores de config.yaml

    retorna:
        dict con DataFrames:
        {
            'historico_equipos': pd.DataFrame,
            'X':                 pd.DataFrame,
            'Y':                 pd.DataFrame,
            'Z':                 pd.DataFrame,
        }
    """
    path = Path(data_path or config['paths']['data_raw'])
    logger.info(f"Iniciando carga de datos desde: {path}")

    data = {
        'historico_equipos': _load_csv('historico_equipos', path),
        'X':                 _load_csv('X',                 path),
        'Y':                 _load_csv('Y',                 path),
        'Z':                 _load_csv('Z',                 path),
    }

    logger.info("Archivos cargados correctamente")
    for name, df in data.items():
        logger.info(f"  {name}: {df.shape}")

    return data