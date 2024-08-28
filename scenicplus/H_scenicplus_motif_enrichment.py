#!/usr/bin/env python

import os
import sys
import pickle
import tempfile
import logging as log
import warnings
from pathlib import Path
import pyranges as pr
from pycistarget.utils import region_names_to_coordinates
from scenicplus.wrappers.run_pycistarget import run_pycistarget

warnings.simplefilter(action="ignore", category=FutureWarning)
log.basicConfig(level=log.DEBUG)

###########################################

rankings_db = "/path/to/regions_vs_motifs_ranking.feather"  # for cisTarget analysis
scores_db = "/path/to/regions_vs_motifs_scores.feather"  # for DEM analysis
motif_annotation = "/path/to/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
work_dir = Path("/path/to/work_dir")
n_cores = 8

tmp_dir = tempfile.mkdtemp()

###########################################

log.info(f"tmp dir: {tmp_dir}")

atac_path = work_dir / "scATAC"
motifs_path = work_dir / "motifs"
candidate_enhancer_path = atac_path / "candidate_enhancers"

if not os.path.exists(os.path.join(work_dir, "scATAC")):
    os.makedirs(os.path.join(work_dir, "scATAC"))

if not os.path.exists(str(motifs_path)):
    os.makedirs(str(motifs_path))


log.info("load files...")

log.info(f"load {candidate_enhancer_path / 'region_bin_topics_otsu.pkl'}")
with open(candidate_enhancer_path / "region_bin_topics_otsu.pkl", "rb") as f:
    region_bin_topics_otsu = pickle.load(f)

log.info(f"load {candidate_enhancer_path / 'region_bin_topics_top3k.pkl'}")
with open(candidate_enhancer_path / "region_bin_topics_top3k.pkl", "rb") as f:
    region_bin_topics_top3k = pickle.load(f)

log.info(f"load {candidate_enhancer_path / 'markers_dict.pkl'}")
with open(candidate_enhancer_path / "markers_dict.pkl", "rb") as f:
    markers_dict = pickle.load(f)

# filter out cell types with empty dataframes
markers_dict = {k: v for k, v in markers_dict.items() if not v.empty}


log.info("create pyranges objects...")

region_sets = {"topics_otsu": {}, "topics_top_3": {}, "DARs": {}}

for topic in region_bin_topics_otsu.keys():
    regions = region_bin_topics_otsu[topic].index[
        region_bin_topics_otsu[topic].index.str.startswith("chr")
    ]  # only keep regions on known chromosomes
    region_sets["topics_otsu"][topic] = pr.PyRanges(
        region_names_to_coordinates(regions)
    )

for topic in region_bin_topics_top3k.keys():
    regions = region_bin_topics_top3k[topic].index[
        region_bin_topics_top3k[topic].index.str.startswith("chr")
    ]  # only keep regions on known chromosomes
    region_sets["topics_top_3"][topic] = pr.PyRanges(
        region_names_to_coordinates(regions)
    )

for DAR in markers_dict.keys():
    regions = markers_dict[DAR].index[
        markers_dict[DAR].index.str.startswith("chr")
    ]  # only keep regions on known chromosomes
    pyr = pr.PyRanges(region_names_to_coordinates(regions))
    if len(pyr) > 0:
        region_sets["DARs"][DAR] = pyr


for key in region_sets.keys():
    log.info(f"{key}: {region_sets[key].keys()}")


log.info("run pyCisTarget")

# monitoring_thread = start_monitoring(seconds_frozen=10, test_interval=100)

try:
    run_pycistarget(
        region_sets=region_sets,
        species="homo_sapiens",
        save_path=str(motifs_path),
        save_partial=True,
        ctx_db_path=rankings_db,
        dem_db_path=scores_db,
        path_to_motif_annotations=motif_annotation,
        run_without_promoters=True,
        n_cpu=n_cores,
        _temp_dir=tmp_dir,
        include_dashboard=False,
        ignore_reinit_error=True,
        annotation_version="v10nr_clust",
    )
except:
    log.error("an error occurred while running pycistarget, trying again without DARs")
    del region_sets["DARs"]
    run_pycistarget(
        region_sets=region_sets,
        species="homo_sapiens",
        save_path=str(motifs_path),
        save_partial=True,
        ctx_db_path=rankings_db,
        dem_db_path=scores_db,
        path_to_motif_annotations=motif_annotation,
        run_without_promoters=True,
        n_cpu=n_cores,
        _temp_dir=tmp_dir,
        include_dashboard=False,
        ignore_reinit_error=True,
        annotation_version="v10nr_clust",
    )


# monitoring_thread.stop()


log.info("all done.")
