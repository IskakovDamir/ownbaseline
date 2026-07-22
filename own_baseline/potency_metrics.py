"""
potency_metrics.py
==================
Единая reference-имплементация четырёх потентностных / энтропийных метрик
single-cell на одном AnnData, плюс scaffold-randomization null.

Метрики (per cell):
  (1) StemID    -- Shannon entropy распределения экспрессии клетки
                   (плоская энтропия транскриптома; Grun et al. 2016 Cell Stem Cell)
  (2) CytoTRACE -- gene counts (GC) + Gene Counts Signature (GCS = геом. среднее
                   топ-200 генов, наиболее коррелирующих с GC; Gulati et al. 2020 Science)
                   ВНИМАНИЕ: это прокси (GC + GCS), без NNLS+diffusion сглаживания
                   полного CytoTRACE -- именно это просили.
  (3) SCENT SR  -- signalling entropy rate случайного блуждания по PPI-сети,
                   взвешенного экспрессией (Teschendorff & Enver 2017 Nat Commun)
  (4) CCAT      -- Pearson(expression, node degree); mean-field суррогат SR
                   (Teschendorff et al. 2021 Bioinformatics)

Зависимости: numpy, scipy, anndata. pandas используется только для парсинга STRING
и для финального DataFrame (тянется вместе с anndata). НЕТ torch / torchtext / flash-attn.

Точные формулы и первоисточники -- см. docstring каждой функции.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components


# ----------------------------------------------------------------------------- #
#  Утилиты нормализации
# ----------------------------------------------------------------------------- #
def _rows_dense(X):
    """Вернуть плотный 2D float-массив из dense/sparse матрицы."""
    if sp.issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _get_X(adata, layer=None):
    X = adata.layers[layer] if layer is not None else adata.X
    return X


def library_normalize(X, target_sum=None):
    """
    Library-size нормализация в относительные счёты (линейные, неотрицательные).
    x_norm[c,g] = x[c,g] / sum_g x[c,g] * target_sum.
    target_sum=None  ->  медиана суммарных счётов по клеткам (как scanpy по умолчанию).
    Работает и для sparse, и для dense; возвращает плотный массив.
    """
    Xd = _rows_dense(X)
    totals = Xd.sum(axis=1)
    totals_safe = np.where(totals > 0, totals, 1.0)
    if target_sum is None:
        target_sum = np.median(totals[totals > 0]) if np.any(totals > 0) else 1.0
    return Xd / totals_safe[:, None] * target_sum


def _xlogx(x):
    """x*log(x) поэлементно, с 0*log0 = 0 (натуральный логарифм)."""
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = x[pos] * np.log(x[pos])
    return out


# ----------------------------------------------------------------------------- #
#  (1) StemID -- Shannon entropy экспрессии
# ----------------------------------------------------------------------------- #
def stemid_entropy(adata, layer=None, base=2.0, normalize_total=False):
    """
    StemID-потентность = Shannon entropy распределения экспрессии клетки.
    Для клетки c: p_g = x_g / sum_g x_g ;  H_c = -sum_g p_g log_base p_g.

    Реализация через тождество, чтобы не материализовать p:
        H_c = [ log(S_c) - (1/S_c) * sum_g x_g log x_g ] / log(base),   S_c = sum_g x_g.

    Это и есть метод-baseline ("плоская" энтропия транскриптома), который в
    бенчмарках SCENT/CCAT отстаёт -- три остальных метрики намеренно НЕ равны ему.
    Источник: Grun et al. 2016 Cell Stem Cell (StemID); описан как baseline в
    Teschendorff et al. 2021 Bioinformatics.

    normalize_total=False: энтропия инвариантна к library-size масштабу, поэтому
    нормализация не меняет результат -- держим X как есть.
    """
    X = _get_X(adata, layer)
    Xd = _rows_dense(X)
    S = Xd.sum(axis=1)
    S_safe = np.where(S > 0, S, 1.0)
    xlx = _xlogx(Xd).sum(axis=1)           # sum_g x_g log x_g
    H_nat = np.log(S_safe) - xlx / S_safe  # натуральные наты
    H = H_nat / np.log(base)
    H[S <= 0] = 0.0
    return H


# ----------------------------------------------------------------------------- #
#  (2) CytoTRACE-прокси -- gene counts + GCS
# ----------------------------------------------------------------------------- #
def _pearson_cols_vs_vec(M, v):
    """
    Pearson-корреляция каждого столбца M (C x G) с вектором v (длины C).
    Возвращает r длины G; столбцы с нулевой дисперсией -> 0.
    """
    Mc = M - M.mean(axis=0, keepdims=True)
    vc = v - v.mean()
    num = Mc.T @ vc                                   # (G,)
    den = np.sqrt((Mc ** 2).sum(axis=0)) * np.sqrt((vc ** 2).sum())
    r = np.zeros_like(num)
    ok = den > 0
    r[ok] = num[ok] / den[ok]
    return r


def cytotrace_proxy(adata, layer=None, n_top=200, target_sum=None):
    """
    CytoTRACE-прокси (Gulati et al. 2020 Science 367:405; doi 10.1126/science.aax0249).

    Шаги (без финального NNLS+diffusion сглаживания -- это прокси GC + GCS):
      gene counts:  GC_c = #{g : x[c,g] > 0}     (число детектируемых генов)
      нормализация: rel = library_normalize(X);  logn = log1p(rel)   (нат. лог)
      отбор генов:  r_g = Pearson(logn[:,g], GC) по клеткам;  T = top-n_top по r_g
      GCS:          GCS_c = геом. среднее экспрессии топ-генов в клетке
                          = exp( mean_{g in T} logn[c,g] )   (геом. среднее (1+rel))

    Что "сохраняется между датасетами": ранговый порядок (больше
    транскрипционного разнообразия -> ниже дифференцировка). Набор T
    пересчитывается на каждом датасете -- переносится сам монотонный закон.

    Возвращает dict: {'gene_counts', 'gcs', 'gcs_gene_idx', 'gc_corr'}.
    """
    X = _get_X(adata, layer)
    Xd = _rows_dense(X)
    gc = (Xd > 0).sum(axis=1).astype(np.float64)      # GC_c
    rel = library_normalize(Xd, target_sum=target_sum)
    logn = np.log1p(rel)
    r = _pearson_cols_vs_vec(logn, gc)                # корреляция гена с GC
    G = logn.shape[1]
    k = int(min(n_top, G))
    top = np.argsort(r)[-k:]                           # топ-k по положит. корреляции
    gcs = np.exp(logn[:, top].mean(axis=1))            # геом. среднее
    return {"gene_counts": gc, "gcs": gcs, "gcs_gene_idx": top, "gc_corr": r}


# ----------------------------------------------------------------------------- #
#  PPI: загрузка STRING и построение adjacency
# ----------------------------------------------------------------------------- #
def load_string_ppi(links_path, info_path, score_threshold=700):
    """
    Парсит файлы STRING v12.0 (human, taxon 9606) в edge list по символам генов.

    Ссылки на файлы (хост загрузок STRING v12 = stringdb-downloads.org):
      links: 9606.protein.links.v12.0.txt.gz
             https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
      info:  9606.protein.info.v12.0.txt.gz
             https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz
      (страница: https://string-db.org/cgi/download?species_text=Homo+sapiens)

    Формат links: 'protein1 protein2 combined_score' (пробел-разделитель),
    combined_score умножен на 1000 (целое). High-confidence = combined_score >= 700.
    Формат info: '#string_protein_id  preferred_name  protein_size  annotation' (TAB),
    даёт отображение STRING-ID (9606.ENSP...) -> символ гена.

    score_threshold: порог combined_score (700 = high, 900 = highest, 400 = medium).
    Возвращает np.ndarray формы (E, 2) с символами генов (без петель, без дублей).
    """
    import pandas as pd
    info = pd.read_csv(info_path, sep="\t", compression="infer",
                       usecols=[0, 1], names=["sid", "symbol"], header=0)
    id2sym = dict(zip(info["sid"], info["symbol"]))

    links = pd.read_csv(links_path, sep=" ", compression="infer",
                        usecols=["protein1", "protein2", "combined_score"])
    links = links[links["combined_score"] >= score_threshold]
    a = links["protein1"].map(id2sym)
    b = links["protein2"].map(id2sym)
    keep = a.notna() & b.notna() & (a != b)
    a_arr = a[keep].to_numpy()
    b_arr = b[keep].to_numpy()
    swap = a_arr > b_arr                       # каноничный порядок пары (A,B)/(B,A)
    lo = np.where(swap, b_arr, a_arr)
    hi = np.where(swap, a_arr, b_arr)
    de = pd.DataFrame({"lo": lo, "hi": hi}).drop_duplicates()
    return de.to_numpy()


def build_ppi_adjacency(edges, var_names):
    """
    Из edge list (символы генов) и adata.var_names строит:
      A         : scipy.sparse 0/1 симметричная матрица смежности (LCC),
      genes     : список генов в порядке строк/столбцов A,
      col_idx   : np.ndarray -- позиции этих генов в adata (для среза adata.X),
      degree    : вектор степеней k_i = #соседей.

    Берётся пересечение генов сети и adata, затем НАИБОЛЬШАЯ СВЯЗНАЯ КОМПОНЕНТА
    (как в SCENT: 'maximally connected subnetwork').
    """
    var_names = np.asarray(var_names)
    present = set(var_names.tolist())
    # фильтр рёбер на гены, имеющиеся в adata
    mask = np.array([(e0 in present) and (e1 in present) for e0, e1 in edges])
    E = edges[mask]
    if len(E) == 0:
        raise ValueError("Нет пересечения генов PPI-сети и adata.var_names.")

    # локальная индексация по уникальным генам рёбер
    uniq = np.unique(E)
    g2i = {g: i for i, g in enumerate(uniq)}
    n = len(uniq)
    rows = np.array([g2i[e0] for e0, e1 in E])
    cols = np.array([g2i[e1] for e0, e1 in E])
    data = np.ones(len(E))
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = A + A.T
    A = (A > 0).astype(np.float64).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    # наибольшая связная компонента
    ncomp, labels = connected_components(A, directed=False)
    if ncomp > 1:
        sizes = np.bincount(labels)
        big = np.argmax(sizes)
        sel = np.where(labels == big)[0]
        A = A[sel][:, sel]
        uniq = uniq[sel]

    # отображение генов LCC -> столбцы adata
    name2col = {g: i for i, g in enumerate(var_names)}
    col_idx = np.array([name2col[g] for g in uniq])
    degree = np.asarray(A.sum(axis=1)).ravel()
    return A.tocsr(), list(uniq), col_idx, degree


# ----------------------------------------------------------------------------- #
#  (3) SCENT signalling entropy rate SR
# ----------------------------------------------------------------------------- #
def _max_entropy_rate(A):
    """
    Максимальная entropy rate сети = log(lambda_max(A)) (мера Парри / MERW).
    Используется SCENT для нормализации SR в [0,1].
    """
    lam = eigsh(A.astype(np.float64), k=1, which="LA",
                return_eigenvectors=False)[0]
    return float(np.log(lam))


def scent_sr(adata, A, col_idx, layer=None, target_sum=None, chunk=2000):
    """
    Signalling entropy rate SR (Teschendorff & Enver 2017 Nat Commun 8:15599;
    doi 10.1038/ncomms15599; фундамент: Banerji 2013 Sci Rep 3:3039).

    Конструкция (mass action) на PPI-сети с матрицей смежности A (0/1, симметрична):
        вес ребра     w_ij ~ x_i x_j
        стох. матрица p_ij = A_ij x_j / sum_k A_ik x_k          (случайное блуждание)
        лок. энтропия S_i  = -sum_j p_ij log p_ij
        стационар. распр. pi_i = x_i (A x)_i / (x^T A x)        (detailed balance)
        SR(клетка)         = (sum_i pi_i S_i) / log(lambda_max(A))   in [0,1]

    Замкнутая форма для S_i (выводится подстановкой p_ij):
        d_i = (A x)_i ;  S_i = log(d_i) - (A (x ⊙ log x))_i / d_i   (при d_i>0, иначе 0)

    Векторизовано по клеткам блоками (chunk) c sparse-матумножением X@A.
    Экспрессия -- ЛИНЕЙНАЯ нормализованная (SCENT требует положительных значений).

    "Сохраняется между датасетами": обратный закон SR<->потентность; фикс. scaffold = A.
    """
    X = _get_X(adata, layer)
    Xfull = library_normalize(X, target_sum=target_sum)
    Xn = Xfull[:, col_idx]                              # клетки x гены_сети (линейные)
    C = Xn.shape[0]
    sr_max = _max_entropy_rate(A)
    SR = np.empty(C, dtype=np.float64)

    for s in range(0, C, chunk):
        Xc = Xn[s:s + chunk]                            # (c, G)
        D = Xc @ A                                      # d_i для каждой клетки: (c, G)
        D = np.asarray(D)
        XLX = _xlogx(Xc)                                # (c, G)
        AXLX = np.asarray(XLX @ A)                      # sum_j A_ij x_j log x_j
        Dsafe = np.where(D > 0, D, 1.0)
        Smat = np.log(Dsafe) - AXLX / Dsafe             # S_i (там, где D>0)
        piXD = Xc * D                                   # x_i d_i  ~ pi_i (до нормировки)
        Z = piXD.sum(axis=1)                            # x^T A x
        term = piXD * Smat                              # x_i d_i S_i
        term = np.where(D > 0, term, 0.0)               # маска d_i=0 (0*-inf -> 0)
        Zsafe = np.where(Z > 0, Z, 1.0)
        sr = term.sum(axis=1) / Zsafe / sr_max
        sr[Z <= 0] = np.nan
        SR[s:s + chunk] = sr
    return SR


# ----------------------------------------------------------------------------- #
#  (4) CCAT -- Pearson(expression, node degree)
# ----------------------------------------------------------------------------- #
def ccat(adata, col_idx, degree, layer=None, target_sum=None, use_log=True):
    """
    CCAT = Correlation of Connectome And Transcriptome
    (Teschendorff et al. 2021 Bioinformatics 37(11):1528; doi 10.1093/bioinformatics/btaa987).

    Для клетки c: CCAT_c = Pearson по генам сети между вектором экспрессии x_c и
    вектором степеней k (k_i = число соседей в PPI). Значение в [-1, 1].
    Это НЕ энтропия: mean-field приближение SR (локальные S_i почти константны,
    pi_i ~ x_i k_i  =>  SR ~ x·k  =>  нормировка = Pearson(x, k)).

    use_log=True: log1p(rel)-экспрессия (как обычно в LandSCENT). Корреляция
    инвариантна к масштабу клетки, но к log-преобразованию -- нет.
    """
    X = _get_X(adata, layer)
    Xfull = library_normalize(X, target_sum=target_sum)
    Xn = Xfull[:, col_idx]
    if use_log:
        Xn = np.log1p(Xn)
    return _ccat_from_matrix(Xn, degree)


def _ccat_from_matrix(Xn, degree):
    """Ядро CCAT: Pearson каждой строки Xn (клетки x гены) с вектором degree."""
    k = np.asarray(degree, dtype=np.float64)
    kc = k - k.mean()
    kc_norm = np.sqrt((kc ** 2).sum())
    Rc = Xn - Xn.mean(axis=1, keepdims=True)
    num = Rc @ kc
    den = np.sqrt((Rc ** 2).sum(axis=1)) * kc_norm
    out = np.zeros(Xn.shape[0], dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


# ----------------------------------------------------------------------------- #
#  Единый расчёт всех метрик
# ----------------------------------------------------------------------------- #
def compute_all_metrics(adata, ppi_edges=None, layer=None, n_top=200,
                        target_sum=None, ccat_use_log=True):
    """
    Считает все четыре метрики на одном AnnData и возвращает pandas.DataFrame
    (index = adata.obs_names) со столбцами:
        stemid, gene_counts, gcs, scent_sr, ccat
    SCENT/CCAT считаются только если передан ppi_edges (edge list символов генов).
    Возвращает (df, ctx), где ctx -- словарь с A, col_idx, degree, sr_max и пр.
    для последующего scaffold-null.
    """
    import pandas as pd
    out = {}
    out["stemid"] = stemid_entropy(adata, layer=layer)
    cyto = cytotrace_proxy(adata, layer=layer, n_top=n_top, target_sum=target_sum)
    out["gene_counts"] = cyto["gene_counts"]
    out["gcs"] = cyto["gcs"]

    ctx = {"cyto": cyto}
    if ppi_edges is not None:
        A, genes, col_idx, degree = build_ppi_adjacency(ppi_edges, adata.var_names)
        out["scent_sr"] = scent_sr(adata, A, col_idx, layer=layer, target_sum=target_sum)
        out["ccat"] = ccat(adata, col_idx, degree, layer=layer,
                           target_sum=target_sum, use_log=ccat_use_log)
        ctx.update({"A": A, "genes": genes, "col_idx": col_idx,
                    "degree": degree, "ccat_use_log": ccat_use_log,
                    "target_sum": target_sum, "layer": layer})

    df = pd.DataFrame(out, index=np.asarray(adata.obs_names))
    return df, ctx


# ----------------------------------------------------------------------------- #
#  Scaffold-randomization null
# ----------------------------------------------------------------------------- #
def scaffold_null(adata, ctx, score_fn, n_perm=200, seed=0,
                  which=("ccat", "scent_sr", "cytotrace")):
    """
    Scaffold-randomization null: измеряет, какая доля сигнала метрики держится
    НА ФИКСИРОВАННОМ SCAFFOLD, а не на биологии. Для каждой пермутации scaffold
    рандомизируется, метрика пересчитывается, и к ней применяется score_fn.

    Рандомизация по метрикам:
      ccat       : перестановка вектора степеней k (какой ген имеет какую степень).
                   Топология/распределение степеней сохранены, рвётся связь
                   экспрессия<->степень. Точный аналог "permute degree vector".
      scent_sr   : перестановка назначения генов на узлы сети (X_net[:, perm], A фикс.).
                   Сохраняет всю топологию и степени, рвёт соответствие
                   ген<->его реальное сетевое окружение. Это SR-корректный аналог
                   рандомизации scaffold (для SR нельзя «переставить только степень»,
                   т.к. SR использует всю матрицу A, а не только k).
      cytotrace  : перестановка per-cell вектора gene counts по клеткам (рангов GC),
                   с пересчётом набора топ-200 и GCS. Изолирует, сколько сигнала
                   GCS держится на реальном ранжировании клеток по числу генов.

    score_fn(values: np.ndarray[n_cells]) -> float
        пользовательская статистика диагностики (напр. Spearman метрики с известной
        потентностной меткой, или кросс-датасетное согласие). Должна принимать
        вектор значений метрики per cell и возвращать скаляр.

    Возвращает dict: metric -> {'real': float, 'null': np.ndarray[n_perm], 'p_value': float}
    p_value -- двусторонний эмпирический: доля |null| >= |real - mean(null)| ... здесь
    берём правосторонний по |отклонению| (см. ниже). Интерпретация: высокий real при
    низком null => сигнал НЕ держится на scaffold (биология); real внутри null =>
    «сохраняющаяся структура» воспроизводится случайным scaffold (тревога для null).
    """
    rng = np.random.default_rng(seed)
    res = {}

    # ---- общие подготовленные представления ----
    layer = ctx.get("layer", None)
    target_sum = ctx.get("target_sum", None)

    # CCAT real + null
    if "ccat" in which and "col_idx" in ctx:
        col_idx, degree = ctx["col_idx"], ctx["degree"]
        Xfull = library_normalize(_get_X(adata, layer), target_sum=target_sum)
        Xn = Xfull[:, col_idx]
        if ctx.get("ccat_use_log", True):
            Xn = np.log1p(Xn)
        real_vals = _ccat_from_matrix(Xn, degree)
        real = score_fn(real_vals)
        null = np.empty(n_perm)
        for i in range(n_perm):
            kperm = degree[rng.permutation(len(degree))]
            null[i] = score_fn(_ccat_from_matrix(Xn, kperm))
        res["ccat"] = _summ(real, null)

    # SCENT SR real + null
    if "scent_sr" in which and "A" in ctx:
        A, col_idx = ctx["A"], ctx["col_idx"]
        Xfull = library_normalize(_get_X(adata, layer), target_sum=target_sum)
        Xn = Xfull[:, col_idx]
        sr_max = _max_entropy_rate(A)
        real_vals = _sr_from_matrix(Xn, A, sr_max)
        real = score_fn(real_vals)
        null = np.empty(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(Xn.shape[1])
            null[i] = score_fn(_sr_from_matrix(Xn[:, perm], A, sr_max))
        res["scent_sr"] = _summ(real, null)

    # CytoTRACE GCS real + null
    if "cytotrace" in which:
        Xd = _rows_dense(_get_X(adata, layer))
        gc = (Xd > 0).sum(axis=1).astype(np.float64)
        logn = np.log1p(library_normalize(Xd, target_sum=target_sum))
        n_top = len(ctx["cyto"]["gcs_gene_idx"]) if "cyto" in ctx else 200
        real_vals = _gcs_from_gc(logn, gc, n_top)
        real = score_fn(real_vals)
        null = np.empty(n_perm)
        for i in range(n_perm):
            gc_perm = gc[rng.permutation(len(gc))]
            null[i] = score_fn(_gcs_from_gc(logn, gc_perm, n_top))
        res["cytotrace"] = _summ(real, null)

    return res


def _sr_from_matrix(Xn, A, sr_max, chunk=2000):
    """Ядро SR из готовой линейной матрицы клетки x гены_сети."""
    C = Xn.shape[0]
    SR = np.empty(C)
    for s in range(0, C, chunk):
        Xc = Xn[s:s + chunk]
        D = np.asarray(Xc @ A)
        XLX = _xlogx(Xc)
        AXLX = np.asarray(XLX @ A)
        Dsafe = np.where(D > 0, D, 1.0)
        Smat = np.log(Dsafe) - AXLX / Dsafe
        piXD = Xc * D
        Z = piXD.sum(axis=1)
        term = np.where(D > 0, piXD * Smat, 0.0)
        Zsafe = np.where(Z > 0, Z, 1.0)
        sr = term.sum(axis=1) / Zsafe / sr_max
        sr[Z <= 0] = np.nan
        SR[s:s + chunk] = sr
    return SR


def _gcs_from_gc(logn, gc, n_top):
    """GCS из готовой logn-матрицы и (возможно переставленного) вектора gc."""
    r = _pearson_cols_vs_vec(logn, gc)
    k = int(min(n_top, logn.shape[1]))
    top = np.argsort(r)[-k:]
    return np.exp(logn[:, top].mean(axis=1))


def _summ(real, null):
    """Сводка: real, null-распределение, правосторонний эмпирический p по |отклонению|."""
    null = np.asarray(null)
    center = np.nanmean(null)
    dev_real = abs(real - center)
    dev_null = np.abs(null - center)
    p = (np.sum(dev_null >= dev_real) + 1) / (len(null) + 1)
    return {"real": float(real), "null": null, "p_value": float(p),
            "null_mean": float(center), "null_std": float(np.nanstd(null))}


# ----------------------------------------------------------------------------- #
#  Демонстрация + самопроверка (синтетические AnnData и PPI)
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    import anndata as ad
    import pandas as pd
    from scipy.stats import spearmanr

    rng = np.random.default_rng(42)
    n_cells, n_genes = 400, 600

    # --- синтетический «потентностный градиент» ---
    # latent потентность t in [0,1]; высокая t => больше экспрессируемых генов
    # (шире транскриптом) и слегка выше экспрессия хабов сети.
    t = rng.uniform(0, 1, n_cells)

    # синтетическая PPI: scale-free-ish (Barabasi-подобная) по символам g0..g{n_genes-1}
    deg_target = (rng.zipf(2.0, n_genes)).clip(1, 60)
    edges_list = []
    for gi in range(1, n_genes):
        m = int(min(deg_target[gi], gi))
        partners = rng.choice(gi, size=max(1, m // 1), replace=False)
        for p_ in partners:
            edges_list.append((f"g{gi}", f"g{int(p_)}"))
    edges = np.array(edges_list)              # numpy выведет unicode-dtype (не object)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    # степень каждого гена в этой сети (для генерации зависимости экспрессии от хабов)
    from collections import Counter
    degc = Counter()
    for a_, b_ in edges:
        degc[a_] += 1
        degc[b_] += 1
    gene_names = [f"g{i}" for i in range(n_genes)]
    deg_vec = np.array([degc.get(g, 0) for g in gene_names], dtype=float)
    deg_norm = (deg_vec - deg_vec.mean()) / (deg_vec.std() + 1e-9)

    # счётная матрица: базовый Poisson + (а) больше «включённых» генов при высокой t,
    # (б) хабы чуть выше при высокой t (чтобы CCAT/SR реагировали).
    base_rate = 2.0
    counts = np.zeros((n_cells, n_genes))
    for c in range(n_cells):
        # доля активных генов растёт с t
        active_p = 0.2 + 0.6 * t[c]
        active = rng.random(n_genes) < active_p
        rate = base_rate * (1 + 0.8 * t[c] * deg_norm)   # хабы выше при высокой t
        rate = np.clip(rate, 0.05, None)
        counts[c] = rng.poisson(rate) * active
    counts = counts.astype(np.float64)

    adata = ad.AnnData(X=counts)
    adata.var_names = gene_names
    adata.obs_names = [f"cell{i}" for i in range(n_cells)]
    adata.obs["true_potency"] = t

    print("=== compute_all_metrics ===")
    df, ctx = compute_all_metrics(adata, ppi_edges=edges, n_top=100)
    print(df.head(6).round(4))
    print("\nформа df:", df.shape, "| NaN в SR:", int(np.isnan(df['scent_sr']).sum()))
    print("LCC сети: узлов =", len(ctx["genes"]),
          "| log(lambda_max) =", round(_max_entropy_rate(ctx["A"]), 4))

    print("\n=== Spearman каждой метрики с истинной потентностью t ===")
    for col in ["stemid", "gene_counts", "gcs", "scent_sr", "ccat"]:
        rho = spearmanr(df[col], t, nan_policy="omit").statistic
        print(f"  {col:12s}: rho = {rho:+.3f}")

    print("\n=== scaffold-randomization null ===")
    print("score_fn = |Spearman(метрика, t)| ; n_perm=200")
    score_fn = lambda v: abs(spearmanr(v, t, nan_policy="omit").statistic)
    nullres = scaffold_null(adata, ctx, score_fn, n_perm=200, seed=1)
    for m, d in nullres.items():
        print(f"  {m:10s}: real={d['real']:.3f} | "
              f"null={d['null_mean']:.3f}±{d['null_std']:.3f} | p={d['p_value']:.4f}")

    # минимальные инварианты-проверки
    print("\n=== self-checks ===")
    assert df["gene_counts"].min() >= 0
    assert (df["ccat"].abs() <= 1.0 + 1e-9).all(), "CCAT вне [-1,1]"
    finite_sr = df["scent_sr"].dropna()
    assert ((finite_sr >= -1e-9) & (finite_sr <= 1.0 + 1e-6)).all(), "SR вне [0,1]"
    assert df["stemid"].min() >= -1e-9, "энтропия отрицательна"
    print("OK: GC>=0, CCAT in [-1,1], SR in [0,1], StemID>=0")
    print("\nГотово.")
