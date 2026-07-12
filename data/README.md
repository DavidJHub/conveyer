# data/

Coloca aquí los datasets. Rutas que buscan los notebooks:

    data/conversations.parquet            # log crudo de conversaciones (notebooks 01, 03, 04)
    data/similarweb_transformed_data/     # star schema analítico: funnel + topics (notebooks 02, 04)
    data/similarweb_clickstream_data/     # vista conductual: clicks/visitas/compras (clickstream_data_analysis.ipynb)

📖 **Esquemas detallados, diagramas ER y documentación de campos de ambos
datasets SimilarWeb:** [`docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md).

El notebook maestro (`notebooks/04_aces_and_research_questions.ipynb`) usa
los datos si existen y cae a datos sintéticos si faltan. Los experimentos
exportados para el simulador ACES se escriben en `outputs/aces/`.

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
