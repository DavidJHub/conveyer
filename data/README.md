# data/

Coloca aquí el dataset de conversaciones. El notebook
(`notebooks/01_text_clustering_pipeline.ipynb`) busca por defecto:

    data/conversations.parquet

Formatos soportados (autodetectados por extensión): `.parquet`, `.csv`, `.jsonl`, `.json`.

## Columnas esperadas (ajustables en la celda `CONFIG` del notebook)

| Columna                 | Descripción                                              |
|-------------------------|----------------------------------------------------------|
| `question`              | Mensaje del usuario                                      |
| `answer`                | Respuesta del LLM                                        |
| `brands_in_question`    | Lista de marcas mencionadas por el usuario               |
| `brands_in_answer`      | Lista de marcas presentes en la respuesta del LLM        |
| `skin_care_categories`  | Categoría NIQ (taxonomía de referencia para validar)     |
| `session_id`            | Id de conversación (opcional)                            |
| `session_pos`           | Posición del turno en la sesión (opcional; se deriva)    |

Si el archivo no existe, el notebook genera **datos sintéticos** y corre igual,
para que puedas validar el pipeline antes de enchufar los datos reales.

> Los archivos de datos no se versionan (ver `.gitignore`).
