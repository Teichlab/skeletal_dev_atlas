#!/usr/bin/env python

import os
import sys
import pickle
import tempfile
import logging as log
import warnings
from pathlib import Path
import scanpy as sc
from pycisTopic.cistopic_class import *
from pycisTopic.lda_models import *
from pycisTopic.clust_vis import *
from pycisTopic.topic_binarization import *
from pycisTopic.diff_features import *
from utils import load_cistopic_obj, save_cistopic_obj

log.basicConfig(level=log.INFO)
warnings.simplefilter(action="ignore", category=FutureWarning)

###########################################

path_to_blacklist = "/path/to/blacklist.bed"
peak_h5ad = "/path/to/peak.h5ad"
meta_csv = "/path/to/meta.csv"
work_dir = Path("/path/to/work_dir")
cellranger_dir = Path("/path/to/cellranger_dir")
n_cores = 8

tmp_dir = tempfile.mkdtemp()

###########################################


log.info("create output folders")

atac_path = work_dir / "scATAC"
consensus_peak_calling_path = atac_path / "consensus_peak_calling"
qc_path = atac_path / "quality_control"

if not os.path.exists(str(atac_path)):
    os.makedirs(str(atac_path))


############################
#  create cistopic object  #
############################

if peak_h5ad:
    log.info("...load processed ATAC data from h5ad")

    ad_frag = sc.read_h5ad(peak_h5ad)

    frag_mat = ad_frag.X.T.tocsr()
    cell_names = ad_frag.obs_names.tolist()
    region_names = ad_frag.var_names.tolist()

    log.info("create cistopic object...")

    cistopic_obj = create_cistopic_object(
        fragment_matrix=frag_mat,
        cell_names=cell_names,
        region_names=region_names,
        path_to_blacklist=path_to_blacklist,
        path_to_fragments={},
    )
else:
    log.info("load fragments info")

    with open(consensus_peak_calling_path / "fragments_dict.pkl", "rb") as f:
        fragments_dict = pickle.load(f)

    log.info("...load processed ATAC data")

    path_to_regions = {
        sample_id: str(consensus_peak_calling_path / "consensus_regions.bed")
        for sample_id in fragments_dict
    }

    with open(qc_path / "metadata_bc.pkl", "rb") as f:
        metadata_bc = pickle.load(f)

    with open(qc_path / "bc_passing_filters.pkl", "rb") as f:
        bc_passing_filters = pickle.load(f)

    log.info("create cistopic object list per fragment...")

    cistopic_obj_list = [
        create_cistopic_object_from_fragments(
            path_to_fragments=fragments_dict[key],
            path_to_regions=path_to_regions[key],
            path_to_blacklist=path_to_blacklist,
            metrics=metadata_bc[key],
            valid_bc=bc_passing_filters[key],
            n_cpu=n_cores,
            project=key,
        )
        for key in fragments_dict.keys()
    ]

    log.info("...merge cistopic objects")

    cistopic_obj = merge(cistopic_obj_list)


print(cistopic_obj)


##################
#  save to file  #
##################

log.info("save cistopic object")

save_cistopic_obj(cistopic_obj, atac_path / "cistopic_obj.pkl")

log.info("all done.")
