"""Builds notebooks/01_text_clustering_pipeline.ipynb as valid nbformat-4 JSON.

This avoids a hard dependency on `nbformat`. Run: python notebooks/_build_notebook.py
"""
import json
import os

CELLS = []


def md(src: str):
    CELLS.append(("markdown", src.strip("\n") + "\n"))


def code(src: str):
    CELLS.append(("code", src.strip("\n") + "\n"))


# ----------------------------------------------------------------------------
md(r"""
# Clusterización de conversaciones (skincare) — pipeline de segmentación

**Objetivo de este notebook:** obtener *información directa de las conversaciones* — qué
grupos de segmentación existen, cómo se expresan ciertas propiedades dentro de cada grupo,
y el top de recomendaciones que da el LLM cuando se le pide una recomendación.

Está construido sobre la discusión previa, así que separa explícitamente los tres ejes que
suelen mezclarse en una sola clusterización:

1. **Tópico / preocupación** (acné, anti-edad, protección solar, hiperpigmentación…) →
   descubrimiento *data-driven* con embeddings, anclado a la taxonomía NIQ (`skin_care_categories`).
2. **Intención** (informacional / comparación / compra / troubleshooting / rutina) →
   clasificación con esquema explícito (reglas + opcional zero-shot), **no** clustering.
3. **Comportamiento del agente** (¿recomienda marca?, ¿cita fuentes?, ¿una marca o un set?) →
   derivado de las columnas, sin clustering.

> **Honestidad metodológica:** el número de clusters **no** está objetivamente en los datos;
> depende de los hiperparámetros. Por eso el notebook ofrece *dos técnicas fuertes* — BERTopic
> (data-driven, descubre `n`) y KMeans con `n` fijo + barrido de silueta — y reporta estabilidad
> en vez de vender un único `n` como hallazgo.

### Cómo usarlo
- Ajusta `CONFIG` (rutas y nombres de columna) en la celda de configuración.
- Si no hay dataset disponible, el notebook genera **datos sintéticos** realistas y corre de
  punta a punta (útil para validar el pipeline antes de enchufar los datos reales).
- Las dependencias pesadas (embeddings, BERTopic) se degradan con elegancia: si no están
  instaladas, se usa un fallback TF-IDF para que el pipeline no se rompa.
""")

# ----------------------------------------------------------------------------
md(r"""
## 0. Configuración

Mapea aquí los nombres reales de tus columnas. Los defaults siguen la nomenclatura usada en
la conversación.
""")

code(r'''
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    # --- Datos ---
    data_path: str = "../data/conversations.parquet"   # .parquet / .csv / .jsonl (autodetectado)
    use_synthetic_if_missing: bool = True              # corre con datos sintéticos si no existe el archivo
    synthetic_n: int = 1500

    # --- Mapeo de columnas (ajusta a tu esquema real) ---
    col_question: str = "question"
    col_answer: str = "answer"
    col_brands_q: str = "brands_in_question"
    col_brands_a: str = "brands_in_answer"
    col_niq: str = "skin_care_categories"     # taxonomía de referencia (ground truth para validar)
    col_session: str = "session_id"           # id de sesión/conversación (opcional)
    col_session_pos: str = "session_pos"      # posición del turno dentro de la sesión (opcional; se deriva si falta)

    # --- Corpus a clusterizar ---
    # "question" -> intención del usuario | "answer" -> comportamiento del LLM | "qa" -> ambos concatenados
    cluster_on: str = "question"
    first_turn_only: bool = True              # solo primer turno = intención limpia (recomendado para tópico/intención)

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-m3"      # alternativa fuerte abierta; o "intfloat/multilingual-e5-large"
    normalize_embeddings: bool = True

    # --- Clustering ---
    n_clusters: int = 12                      # 'n' para KMeans (el usuario pidió 'n grupos')
    k_sweep: tuple = (4, 6, 8, 10, 12, 16, 20)  # barrido para elegir n por silueta
    # BERTopic / HDBSCAN
    min_cluster_size: int = 40
    min_samples: int = 8
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    random_state: int = 42

    # --- LLM naming (opcional) ---
    use_llm_naming: bool = False              # requiere ANTHROPIC_API_KEY
    llm_model: str = "claude-fable-5"

    # --- Salidas ---
    out_dir: str = "../outputs"

CFG = Config()
import os
os.makedirs(CFG.out_dir, exist_ok=True)
CFG
''')

# ----------------------------------------------------------------------------
md(r"""
## 1. Dependencias

Instala lo necesario (descomenta). El pipeline detecta qué hay disponible y se adapta.
""")

code(r'''
# !pip install -q pandas numpy scikit-learn pyarrow
# !pip install -q sentence-transformers bertopic umap-learn hdbscan
# !pip install -q anthropic   # solo si vas a usar naming con LLM

import importlib

def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False

CAPS = {
    "sentence_transformers": _have("sentence_transformers"),
    "bertopic": _have("bertopic"),
    "umap": _have("umap"),
    "hdbscan": _have("hdbscan"),
    "anthropic": _have("anthropic"),
}
print("Capacidades detectadas:")
for k, v in CAPS.items():
    print(f"  {'OK ' if v else '-- '} {k}")
if not CAPS["sentence_transformers"]:
    print("\n[fallback] sin sentence-transformers -> se usará TF-IDF para embeddings.")
if not CAPS["bertopic"]:
    print("[fallback] sin bertopic -> el clustering data-driven usa HDBSCAN/KMeans directo.")
''')

# ----------------------------------------------------------------------------
md(r"""
## 2. Carga de datos (con fallback sintético)

El generador sintético produce conversaciones de skincare plausibles: preguntas con
distintas **intenciones**, marcas mencionadas por el usuario vs. introducidas por el LLM, y
categorías NIQ. Sirve para validar el pipeline; reemplázalo enchufando tu `data_path`.
""")

code(r'''
import pandas as pd
import numpy as np
import ast

def _read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(f"Formato no soportado: {path}")

def make_synthetic(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = {
        "acne": (["acne", "granos", "brotes", "puntos negros"],
                 ["CeraVe", "La Roche-Posay", "The Ordinary", "Paula's Choice"]),
        "anti-aging": (["arrugas", "anti-edad", "lineas finas", "firmeza", "retinol"],
                       ["Olay", "RoC", "The Ordinary", "Drunk Elephant", "Estee Lauder"]),
        "sunscreen": (["protector solar", "spf", "fotoprotector", "fps 50"],
                      ["La Roche-Posay", "ISDIN", "Bioderma", "EltaMD"]),
        "hyperpigmentation": (["manchas", "hiperpigmentacion", "tono desigual", "vitamina c"],
                              ["The Ordinary", "Good Molecules", "Murad", "SkinCeuticals"]),
        "hydration": (["hidratacion", "piel seca", "humectante", "acido hialuronico"],
                      ["CeraVe", "Neutrogena", "Cetaphil", "Vichy"]),
        "sensitive": (["piel sensible", "rojez", "rosacea", "irritacion"],
                      ["Avene", "Bioderma", "La Roche-Posay", "Cetaphil"]),
    }
    intents = {
        "informational": ["que es {t}", "como funciona {t}", "para que sirve {t}", "rutina para {t}"],
        "comparison":    ["{b1} vs {b2} para {t}", "cual es mejor para {t}, {b1} o {b2}", "diferencia entre {b1} y {b2}"],
        "purchase":      ["cual es el mejor producto para {t}", "que me recomiendas para {t}", "donde comprar algo para {t}", "vale la pena {b1} para {t}"],
        "troubleshooting": ["{b1} me esta irritando, que hago", "mi {t} empeoro usando {b1}", "puedo combinar {b1} con retinol"],
        "routine":       ["arma una rutina para {t}", "orden de aplicacion para {t}", "rutina manana y noche para {t}"],
    }
    rows = []
    cat_names = list(cats.keys())
    for i in range(n):
        cat = rng.choice(cat_names)
        terms, brands = cats[cat]
        t = rng.choice(terms)
        intent = rng.choice(list(intents.keys()), p=[0.3, 0.18, 0.27, 0.15, 0.10])
        b1, b2 = rng.choice(brands, size=2, replace=False)
        q = rng.choice(intents[intent]).format(t=t, b1=b1, b2=b2)

        # marcas mencionadas por el usuario en la pregunta
        if intent in ("comparison", "troubleshooting") or rng.random() < 0.25:
            brands_q = [b1] if intent != "comparison" else [b1, b2]
        else:
            brands_q = []
        # marcas en la respuesta: el LLM suele introducir marcas no solicitadas
        n_rec = rng.integers(1, 4)
        brands_a = list(rng.choice(brands, size=min(n_rec, len(brands)), replace=False))
        # asegura algo de endorsement de las de la pregunta
        brands_a = list(dict.fromkeys(brands_q + brands_a))
        cites = rng.random() < 0.4
        ans = f"Para {t}, considera {', '.join(brands_a)}." + (" [fuentes citadas]" if cites else "")

        sess = f"s{i // 3}"  # ~3 turnos por sesion
        rows.append(dict(
            session_id=sess, question=q, answer=ans,
            brands_in_question=brands_q, brands_in_answer=brands_a,
            skin_care_categories=cat, cites_sources=cites, _intent_true=intent,
        ))
    df = pd.DataFrame(rows)
    # session_pos por orden de aparicion dentro de cada sesion
    df["session_pos"] = df.groupby("session_id").cumcount() + 1
    return df

if os.path.exists(CFG.data_path):
    df = _read_any(CFG.data_path)
    print(f"Datos cargados: {CFG.data_path} -> {df.shape}")
elif CFG.use_synthetic_if_missing:
    df = make_synthetic(CFG.synthetic_n, CFG.random_state)
    print(f"[SINTÉTICO] {CFG.data_path} no existe. Generadas {len(df)} filas de prueba.")
else:
    raise FileNotFoundError(CFG.data_path)

df.head(3)
''')

# ----------------------------------------------------------------------------
md(r"""
## 3. Parseo y feature engineering

Normaliza las columnas de marcas (pueden venir como listas reales o como strings tipo
`"['CeraVe', ...]"`), deriva `session_pos` si falta, y construye las features que conectan
con el relato *recomendación → intención de compra*.
""")

code(r'''
def to_list(x):
    """Convierte celdas a lista de strings, tolerando listas reales, stringified o NaN."""
    if isinstance(x, list):
        return [str(b).strip() for b in x if str(b).strip()]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                return [str(b).strip() for b in v if str(b).strip()]
            except Exception:
                pass
        # CSV simple "a, b, c"
        return [t.strip() for t in s.split(",") if t.strip()]
    return []

def norm(s: str) -> str:
    return str(s).strip().lower()

C = CFG
# columnas de marcas -> listas normalizadas
df["_brands_q"] = df.get(C.col_brands_q, pd.Series([[]] * len(df))).apply(to_list)
df["_brands_a"] = df.get(C.col_brands_a, pd.Series([[]] * len(df))).apply(to_list)
df["_brands_q_norm"] = df["_brands_q"].apply(lambda L: set(map(norm, L)))
df["_brands_a_norm"] = df["_brands_a"].apply(lambda L: set(map(norm, L)))

# session_pos si falta
if C.col_session_pos not in df.columns:
    if C.col_session in df.columns:
        df[C.col_session_pos] = df.groupby(C.col_session).cumcount() + 1
    else:
        df[C.col_session_pos] = 1
        print("[aviso] sin columna de sesión ni session_pos: todo se trata como primer turno.")

# texto base para clustering
q = df[C.col_question].fillna("").astype(str)
a = df.get(C.col_answer, pd.Series([""] * len(df))).fillna("").astype(str)
if C.cluster_on == "question":
    df["_text"] = q
elif C.cluster_on == "answer":
    df["_text"] = a
else:
    df["_text"] = (q + " [SEP] " + a).str.strip()

print("brands_in_question (muestra):", df["_brands_q"].head(3).tolist())
print("brands_in_answer (muestra):  ", df["_brands_a"].head(3).tolist())
df[[C.col_question, C.col_session_pos]].head(3)
''')

code(r'''
# --- Evento de recomendación: marcas que el LLM introduce y el usuario NO mencionó ---
df["_unsolicited"] = df.apply(lambda r: r["_brands_a_norm"] - r["_brands_q_norm"], axis=1)
df["_endorsed"]    = df.apply(lambda r: r["_brands_a_norm"] & r["_brands_q_norm"], axis=1)
df["is_recommendation"] = df["_unsolicited"].apply(len) > 0
df["n_brands_answer"]   = df["_brands_a_norm"].apply(len)

print(f"Tasa de recomendación (respuestas con marca no solicitada): {df['is_recommendation'].mean():.1%}")
print(f"Marcas promedio por respuesta: {df['n_brands_answer'].mean():.2f}")
''')

code(r'''
# --- Intención por reglas (esquema explícito, reproducible) ---
import re

INTENT_PATTERNS = {
    "comparison":      [r"\bvs\b", r"\bo\b .*\?", r"mejor.*,", r"cu[aá]l es mejor", r"diferencia entre", r"compar"],
    "purchase":        [r"mejor producto", r"qu[eé] me recomiendas", r"recomi[eé]nda", r"d[oó]nde comprar",
                        r"vale la pena", r"cu[aá]l comprar", r"best ", r"worth it", r"where to buy", r"dupe"],
    "troubleshooting": [r"irrit", r"empeor", r"reacci[oó]n", r"me sali[oó]", r"puedo combinar", r"se puede mezclar", r"efecto secundario"],
    "routine":         [r"rutina", r"orden de aplicaci[oó]n", r"ma[nñ]ana y noche", r"paso a paso", r"routine"],
    "informational":   [r"qu[eé] es", r"c[oó]mo funciona", r"para qu[eé] sirve", r"qu[eé] hace", r"what is", r"how does"],
}
INTENT_PRIORITY = ["comparison", "purchase", "troubleshooting", "routine", "informational"]

def classify_intent(text: str) -> str:
    t = norm(text)
    for intent in INTENT_PRIORITY:
        for pat in INTENT_PATTERNS[intent]:
            if re.search(pat, t):
                return intent
    return "informational"

df["intent"] = df[C.col_question].fillna("").apply(classify_intent)
# señal de intención de compra/recomendación pedida explícitamente
df["asks_recommendation"] = df["intent"].isin(["purchase", "comparison"])

print(df["intent"].value_counts())
if "_intent_true" in df.columns:
    acc = (df["intent"] == df["_intent_true"]).mean()
    print(f"\n[sintético] accuracy reglas vs intención verdadera: {acc:.1%} (solo referencia)")
''')

# ----------------------------------------------------------------------------
md(r"""
## 4. Brand amplification

`amplification = share_en_respuestas / share_en_preguntas`. Arriba de 1 → el LLM empuja la
marca más allá de la demanda orgánica del usuario (actúa como recomendador, no como espejo).
Es la métrica más cercana al relato de ventas **sin** datos de conversión.
""")

code(r'''
from collections import Counter

def brand_counts(series_of_sets):
    c = Counter()
    for s in series_of_sets:
        c.update(s)
    return c

q_counts = brand_counts(df["_brands_q_norm"])
a_counts = brand_counts(df["_brands_a_norm"])
q_total, a_total = sum(q_counts.values()) or 1, sum(a_counts.values()) or 1

brands = sorted(set(q_counts) | set(a_counts))
amp = pd.DataFrame({
    "brand": brands,
    "q_count": [q_counts[b] for b in brands],
    "a_count": [a_counts[b] for b in brands],
})
amp["q_share"] = amp["q_count"] / q_total
amp["a_share"] = amp["a_count"] / a_total
amp["amplification"] = amp["a_share"] / amp["q_share"].replace(0, np.nan)
amp = amp.sort_values("a_count", ascending=False)
print("Top marcas por presencia en respuestas:")
amp.head(20)
''')

# ----------------------------------------------------------------------------
md(r"""
## 5. Embeddings del corpus

Embebemos el corpus elegido (`CONFIG.cluster_on`). Por defecto solo **primer turno**
(`session_pos == 1`) para intención limpia. Si no hay `sentence-transformers`, caemos a
TF-IDF + SVD (suficiente para que el pipeline corra; cambia los embeddings para resultados serios).
""")

code(r'''
# subconjunto a clusterizar
mask = (df[C.col_session_pos] == 1) if C.first_turn_only else pd.Series(True, index=df.index)
work = df[mask].copy().reset_index(drop=True)
texts = work["_text"].tolist()
print(f"Documentos a clusterizar: {len(texts)} (first_turn_only={C.first_turn_only})")

def embed_corpus(texts):
    if CAPS["sentence_transformers"]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(C.embedding_model)
        emb = model.encode(texts, show_progress_bar=True,
                           normalize_embeddings=C.normalize_embeddings)
        return np.asarray(emb), f"sentence-transformers:{C.embedding_model}"
    # fallback TF-IDF + SVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
    X = tfidf.fit_transform(texts)
    k = min(256, X.shape[1] - 1, max(2, X.shape[0] - 1))
    svd = TruncatedSVD(n_components=k, random_state=C.random_state)
    emb = svd.fit_transform(X)
    if C.normalize_embeddings:
        emb = normalize(emb)
    return emb, f"tfidf+svd({k})"

embeddings, emb_name = embed_corpus(texts)
print("Embeddings:", embeddings.shape, "via", emb_name)
''')

# ----------------------------------------------------------------------------
md(r"""
## 6. Clustering — dos técnicas fuertes

- **A) BERTopic / HDBSCAN (data-driven):** descubre `n` y un cluster de ruido (`-1`). Reportamos
  el tamaño del ruido — no lo ocultamos.
- **B) KMeans con `n` fijo + barrido de silueta:** porque pediste *"n grupos"*. La silueta ayuda a
  elegir `n`, pero recuerda: es una elección de hiperparámetro, no un hecho del dataset.
""")

code(r'''
# --- A) Data-driven: BERTopic si está, si no HDBSCAN directo ---
topics_ddriven = None
topic_model = None
if CAPS["bertopic"] and CAPS["umap"] and CAPS["hdbscan"]:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    umap_model = UMAP(n_neighbors=C.umap_n_neighbors, n_components=C.umap_n_components,
                      metric="cosine", random_state=C.random_state)
    hdb = HDBSCAN(min_cluster_size=C.min_cluster_size, min_samples=C.min_samples,
                  metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdb,
                           vectorizer_model=CountVectorizer(stop_words=None, ngram_range=(1, 2)),
                           calculate_probabilities=False, verbose=False)
    topics_ddriven, _ = topic_model.fit_transform(texts, embeddings)
    topics_ddriven = np.asarray(topics_ddriven)
elif CAPS["hdbscan"]:
    import hdbscan
    clu = hdbscan.HDBSCAN(min_cluster_size=C.min_cluster_size, min_samples=C.min_samples,
                          metric="euclidean", cluster_selection_method="eom")
    topics_ddriven = clu.fit_predict(embeddings)
else:
    print("[fallback] sin HDBSCAN/BERTopic -> se omite la vía data-driven; usa KMeans (sección B).")

if topics_ddriven is not None:
    n_topics = len(set(topics_ddriven)) - (1 if -1 in topics_ddriven else 0)
    noise = float((topics_ddriven == -1).mean())
    print(f"BERTopic/HDBSCAN -> {n_topics} clusters | ruido(-1): {noise:.1%}")
    if noise > 0.4:
        print("  [aviso] ruido alto (>40%): min_cluster_size probablemente muy agresivo para texto corto.")
''')

code(r'''
# --- B) KMeans con n fijo + barrido de silueta ---
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sweep = []
for k in C.k_sweep:
    if k >= len(texts):
        continue
    km = KMeans(n_clusters=k, random_state=C.random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    sil = silhouette_score(embeddings, labels, metric="cosine") if len(set(labels)) > 1 else float("nan")
    sweep.append({"k": k, "silhouette": sil, "inertia": km.inertia_})
sweep_df = pd.DataFrame(sweep)
print("Barrido de k (silueta coseno; texto corto homogéneo => valores bajos esperables):")
print(sweep_df.to_string(index=False))

best_k = int(sweep_df.sort_values("silhouette", ascending=False)["k"].iloc[0]) if len(sweep_df) else C.n_clusters
print(f"\nMejor k por silueta: {best_k}  |  n fijo solicitado en CONFIG: {C.n_clusters}")

# clustering final con el n pedido (cambia a best_k si prefieres data-driven dentro de KMeans)
final_k = C.n_clusters
km = KMeans(n_clusters=final_k, random_state=C.random_state, n_init=10)
work["cluster_kmeans"] = km.fit_predict(embeddings)
if topics_ddriven is not None:
    work["cluster_bertopic"] = topics_ddriven

# elige la etiqueta principal para el resto del análisis
PRIMARY = "cluster_kmeans"   # cambia a "cluster_bertopic" si quieres la vía data-driven
work["cluster"] = work[PRIMARY]
print(f"\nEtiqueta principal: {PRIMARY} con {final_k} grupos")
work["cluster"].value_counts().sort_index()
''')

# ----------------------------------------------------------------------------
md(r"""
## 7. Caracterización de clusters — *cómo se expresan las propiedades dentro de cada grupo*

Para cada cluster: tamaño, distribución de intención, tasa de recomendación, nº de marcas por
respuesta, categoría NIQ dominante, top de marcas, y documentos representativos. Esta tabla es
el verdadero entregable: los clusters son los **segmentos** sobre los que se leen las métricas.
""")

code(r'''
def top_brands_for(sub, col="_brands_a_norm", k=5):
    c = brand_counts(sub[col])
    return ", ".join(f"{b}({n})" for b, n in c.most_common(k)) if c else "-"

rows = []
for cl, sub in work.groupby("cluster"):
    intent_mix = sub["intent"].value_counts(normalize=True)
    niq_mode = sub[C.col_niq].astype(str).mode()
    rows.append({
        "cluster": cl,
        "size": len(sub),
        "share": len(sub) / len(work),
        "niq_dominante": niq_mode.iloc[0] if len(niq_mode) else "-",
        "intent_top": intent_mix.index[0] if len(intent_mix) else "-",
        "%purchase": sub["intent"].eq("purchase").mean(),
        "%comparison": sub["intent"].eq("comparison").mean(),
        "reco_rate": sub["is_recommendation"].mean(),
        "marcas/resp": sub["n_brands_answer"].mean(),
        "top_marcas_respuesta": top_brands_for(sub),
    })
cluster_summary = pd.DataFrame(rows).sort_values("size", ascending=False)
pd.set_option("display.max_colwidth", 60)
cluster_summary
''')

code(r'''
# Documentos representativos por cluster (más cercanos al centroide)
from sklearn.metrics.pairwise import cosine_similarity

emb_by_idx = {i: embeddings[i] for i in range(len(embeddings))}
reps = {}
for cl in sorted(work["cluster"].unique()):
    idx = work.index[work["cluster"] == cl].tolist()
    if not idx:
        continue
    centroid = embeddings[idx].mean(axis=0, keepdims=True)
    sims = cosine_similarity(embeddings[idx], centroid).ravel()
    order = np.argsort(-sims)[:5]
    reps[cl] = [texts[idx[o]] for o in order]

for cl, examples in list(reps.items())[:8]:
    print(f"\n=== Cluster {cl} ===")
    for e in examples:
        print("  •", e[:110])
''')

# ----------------------------------------------------------------------------
md(r"""
## 8. Validación contra la taxonomía NIQ

¿Los clusters no supervisados recuperan `skin_care_categories`? ARI/NMI cercanos a 0 →
las conversaciones no respetan la taxonomía de retail (informativo); cercanos a 1 → el
clustering no añadió nada nuevo.
""")

code(r'''
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

if C.col_niq in work.columns:
    niq = work[C.col_niq].astype(str).tolist()
    print(f"KMeans   vs NIQ -> ARI={adjusted_rand_score(niq, work['cluster_kmeans']):.3f} | "
          f"NMI={normalized_mutual_info_score(niq, work['cluster_kmeans']):.3f}")
    if "cluster_bertopic" in work.columns:
        print(f"BERTopic vs NIQ -> ARI={adjusted_rand_score(niq, work['cluster_bertopic']):.3f} | "
              f"NMI={normalized_mutual_info_score(niq, work['cluster_bertopic']):.3f}")
else:
    print("Sin columna NIQ: se omite validación externa.")
''')

# ----------------------------------------------------------------------------
md(r"""
## 9. Top de recomendaciones cuando se pide una recomendación

Filtramos los turnos donde el usuario **pide** recomendación o comparación
(`asks_recommendation`) y miramos qué marcas introduce el LLM (no solicitadas). Global y por
cluster: este es uno de los entregables que pediste explícitamente.
""")

code(r'''
rec_mask = df["asks_recommendation"]
rec = df[rec_mask]
print(f"Turnos donde se pide recomendación/comparación: {len(rec)} ({rec_mask.mean():.1%} del total)\n")

# Top marcas NO solicitadas que el LLM introduce al pedírsele recomendación
unsol = brand_counts(rec["_unsolicited"])
top_unsol = pd.DataFrame(unsol.most_common(20), columns=["brand", "veces_recomendada"])
print("== Top recomendaciones no solicitadas (LLM) cuando se pide recomendación ==")
print(top_unsol.to_string(index=False))
''')

code(r'''
# Mismo top pero por cluster (usa el work de primer turno, que es donde tenemos cluster)
w_rec = work[work["asks_recommendation"]]
per_cluster = []
for cl, sub in w_rec.groupby("cluster"):
    c = brand_counts(sub["_unsolicited"])
    top = ", ".join(f"{b}({n})" for b, n in c.most_common(5)) if c else "-"
    per_cluster.append({"cluster": cl, "n_pide_reco": len(sub),
                        "reco_rate": sub["is_recommendation"].mean(),
                        "top_recomendaciones": top})
print("== Top recomendaciones por cluster (solo turnos que piden recomendación) ==")
pd.DataFrame(per_cluster).sort_values("n_pide_reco", ascending=False)
''')

# ----------------------------------------------------------------------------
md(r"""
## 10. (Opcional) Nombrado de clusters con LLM

En vez de leer keywords c-TF-IDF, le pasamos los documentos representativos a un LLM y pedimos
una etiqueta corta + si el cluster luce *purchase-intent-heavy*. Requiere `ANTHROPIC_API_KEY`.
Si no está configurado, se usa un nombrado por keywords como fallback.
""")

code(r'''
def name_clusters_llm(reps: dict, model: str):
    import anthropic, json as _json
    client = anthropic.Anthropic()
    out = {}
    for cl, examples in reps.items():
        prompt = (
            "Eres analista de research de skincare. Dado un cluster de preguntas de usuarios a "
            "un asistente, devuelve JSON con: label (<=4 palabras), intent_dominante "
            "(informational|comparison|purchase|troubleshooting|routine), purchase_intent_heavy (bool).\n\n"
            "Preguntas:\n- " + "\n- ".join(examples)
        )
        msg = client.messages.create(model=model, max_tokens=200,
                                     messages=[{"role": "user", "content": prompt}])
        txt = msg.content[0].text
        try:
            out[cl] = _json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except Exception:
            out[cl] = {"label": txt[:40], "intent_dominante": "?", "purchase_intent_heavy": None}
    return out

def name_clusters_keywords(work, embeddings, texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    out = {}
    for cl in sorted(work["cluster"].unique()):
        idx = work.index[work["cluster"] == cl].tolist()
        docs = [texts[work.index.get_loc(i)] for i in idx] if False else [texts[j] for j, _ in enumerate(texts) if work.iloc[j]["cluster"] == cl]
        if not docs:
            continue
        try:
            v = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
            X = v.fit_transform(docs)
            scores = np.asarray(X.mean(axis=0)).ravel()
            terms = np.array(v.get_feature_names_out())
            top = terms[np.argsort(-scores)[:4]]
            out[cl] = {"label": " / ".join(top), "intent_dominante": work[work["cluster"] == cl]["intent"].mode().iloc[0]}
        except Exception:
            out[cl] = {"label": f"cluster {cl}"}
    return out

if CFG.use_llm_naming and CAPS["anthropic"] and os.environ.get("ANTHROPIC_API_KEY"):
    names = name_clusters_llm(reps, CFG.llm_model)
else:
    names = name_clusters_keywords(work, embeddings, texts)
    if CFG.use_llm_naming:
        print("[fallback] sin anthropic/API key -> nombrado por keywords.")

for cl, info in names.items():
    print(f"Cluster {cl}: {info}")
''')

# ----------------------------------------------------------------------------
md(r"""
## 11. Exportar resultados
""")

code(r'''
# Tabla de filas etiquetadas (primer turno) + resúmenes
labeled = work[[C.col_question, "cluster", "cluster_kmeans", "intent",
                "asks_recommendation", "is_recommendation", "n_brands_answer", C.col_niq]].copy()
if "cluster_bertopic" in work.columns:
    labeled["cluster_bertopic"] = work["cluster_bertopic"]
labeled["cluster_name"] = labeled["cluster"].map(lambda c: names.get(c, {}).get("label", str(c)))

labeled.to_csv(os.path.join(CFG.out_dir, "labeled_first_turn.csv"), index=False)
cluster_summary.to_csv(os.path.join(CFG.out_dir, "cluster_summary.csv"), index=False)
amp.to_csv(os.path.join(CFG.out_dir, "brand_amplification.csv"), index=False)
top_unsol.to_csv(os.path.join(CFG.out_dir, "top_recommendations.csv"), index=False)
print("Guardado en", os.path.abspath(CFG.out_dir))
print(os.listdir(CFG.out_dir))
''')

# ----------------------------------------------------------------------------
md(r"""
## Notas y limitaciones (no las escondas en el reporte)

- **El `n` no es un hallazgo.** Reporta el barrido de silueta y, si puedes, estabilidad entre
  semillas/parámetros. Un único run no demuestra "el dataset tiene N tópicos".
- **Texto corto y homogéneo** (todo skincare, queries <15 palabras): siluetas bajas y clusters
  solapados son esperables. No sobrevendas la separación.
- **Ruido de HDBSCAN (`-1`)**: inspecciónalo, no lo dropees en silencio.
- **Intención por reglas** es un punto de partida; para producción usa zero-shot/LLM con el
  esquema explícito y valida sobre una muestra etiquetada a mano.
- **`first_turn_only`** da intención limpia pero descarta el funnel. El siguiente paso natural es
  el análisis **a nivel de sesión** (categoría → marca → intención de compra) como proxy de embudo.
- **Sin conversión real ni demografía**: el clustering es descriptivo. Las métricas
  (reco_rate, amplification, top recomendaciones) son proxies del relato recomendación→ventas,
  no causalidad.
""")

# ----------------------------------------------------------------------------
nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s} if t == "markdown" else
        {"cell_type": t, "metadata": {}, "execution_count": None, "outputs": [], "source": s}
        for (t, s) in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(__file__), "01_text_clustering_pipeline.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Escrito:", out_path, "con", len(CELLS), "celdas")
