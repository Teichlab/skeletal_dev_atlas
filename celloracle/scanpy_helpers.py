"""
single cell analysis helper functions
"""

import gc
import logging as log
from collections import namedtuple
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import scanpy as sc
import anndata as adat
from sknetwork.hierarchy import Paris, LouvainHierarchy, cut_balanced


def graph_smooth(expression_matrix, neighbor_matrix):
    """smoothing over the knn graph"""
    return ((neighbor_matrix @ expression_matrix) + expression_matrix) / (
        neighbor_matrix > 0
    ).sum(axis=1)


def aggregate_pseudobulks(
    expression_matrix,
    neighbor_matrix,
    max_group_size=10,
    method="Paris",
    summarise="mean",
):
    """
    Divide observations by knn graph into clusters of a
    maximum size and return summed expression per cluster.
    If 'min_group_size', add smaller clusters to closest
    cluster (note: this may result in clusters larger than
    'max_group_size').
    Options for method: ['Paris', 'Louvain']
    Options for summarise: ['mean', 'sum']
    """
    Result = namedtuple("Result", ["agg_mat", "groups"])

    # check input size
    n_obs = neighbor_matrix.shape[0]
    if n_obs < max_group_size:
        warnings.warn(
            "n_obs is smaller than max_group_size: "
            f"{n_obs} < {max_group_size}. "
            f"Setting max_group_size to {n_obs}."
        )
        max_group_size = n_obs  # TODO: better choice?

    # create a dendrogram
    if method == "Paris":
        paris = Paris()
        dendrogram = paris.fit_transform(neighbor_matrix)
    elif method == "Louvain":
        louvain = LouvainHierarchy()
        dendrogram = louvain.fit_transform(neighbor_matrix)
    else:
        raise ValueError(f"method '{method}' unknown")

    # cut into groups
    groups = cut_balanced(dendrogram, max_group_size)

    # aggregate
    cut_mat = np.zeros((np.unique(groups).size, expression_matrix.shape[0]))
    for i, g in enumerate(pd.unique(groups)):
        cut_mat[i, :] = groups == g

    # summarise expression by group
    if summarise == "mean":
        pb_mat = ((cut_mat @ expression_matrix).T / cut_mat.sum(axis=1)).T
    elif summarise == "sum":
        pb_mat = cut_mat @ expression_matrix
    else:
        raise ValueError(f"option '{summarise}' for summarise unknown")

    return Result(pb_mat, groups)


def aggregate_meta_df(meta_df):
    def agg_func(x):
        try:
            if is_numeric_dtype(x):
                return lambda y: y.mean()
            else:
                return lambda y: y.value_counts().index[0]
        except:
            return lambda y: np.nan

    idx_name = meta_df.index.name or "index"
    agg_df = meta_df.reset_index(names=idx_name)
    agg_df = (
        agg_df.groupby("metacell", as_index=False)
        .agg({k: agg_func(v) for k, v in agg_df.items()})
        .set_index(idx_name)
    )

    return agg_df


def aggregate_counts(adata, min_cells=0, **kwargs):
    n_mat = adata.obsp["connectivities"].copy()
    agg_res = aggregate_pseudobulks(adata.X, n_mat, **kwargs)

    group_count = np.bincount(agg_res.groups)
    drop_groups = [i for i, c in enumerate(group_count) if c < min_cells]
    log.debug(f"#metacells per size: {np.bincount(group_count)}")

    Result = namedtuple("Result", ["agg_mat", "groups", "drop_groups"])
    return Result(agg_res.agg_mat, agg_res.groups, drop_groups)


def get_metacells(adata, target_n_metacells=None, **kwargs):
    n_cells = adata.shape[0]

    if target_n_metacells:
        kwargs["max_group_size"] = max(1, round(n_cells / target_n_metacells))
        log.info(f"aiming for metacell size: {kwargs['max_group_size']}")
        if "min_cells" in kwargs and kwargs["min_cells"] > kwargs["max_group_size"]:
            log.warning(f"`min_cells` > `max_group_size`, setting `min_cells = 0`")
            kwargs["min_cells"] = 0

    if "max_group_size" not in kwargs or kwargs["max_group_size"] > 1:
        # aggregate counts
        agg_res = aggregate_counts(adata, **kwargs)

        obs_df = adata.obs.assign(metacell=agg_res.groups)
        metacell_order = obs_df.metacell.drop_duplicates().index
        # metacell_bc = obs_df["metacell"].to_dict()

        # aggregate meta information
        obs_df = aggregate_meta_df(obs_df)
        obs_df = obs_df.reindex(metacell_order)

        ad_out = sc.AnnData(
            agg_res.agg_mat, var=adata.var, obs=obs_df, dtype=agg_res.agg_mat.dtype
        )
        if agg_res.drop_groups:
            log.debug(f"dropping {len(agg_res.drop_groups)} metacells")
            ad_out = ad_out[~ad_out.obs.metacell.isin(agg_res.drop_groups)]

        return ad_out
    else:
        adata.obs["metacell"] = list(range(n_cells))
        return adata


def get_metacells_by_group(adata, annot, compute_neighbors=True, **kwargs):
    adatas = []
    groups = adata.obs[annot].unique().tolist()
    for i, cat in enumerate(groups):
        log.info(f"create metacells for group '{cat}' ... ({i+1}/{len(groups)})")
        gc.collect()

        ad = adata[adata.obs[annot] == cat]

        if compute_neighbors:
            sc.pp.pca(ad)
            sc.pp.neighbors(ad)

        adatas.append(get_metacells(ad, **kwargs))
    return adat.concat(adatas, index_unique=None)
