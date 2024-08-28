#!/usr/bin/env python

import os
import json
import dill
import tempfile
import gc
import logging as log
import warnings
from pathlib import Path
import numpy as np
import scanpy as sc
import pybiomart as pbm
from scenicplus.scenicplus_class import create_SCENICPLUS_object
from scenicplus.wrappers.run_scenicplus import run_scenicplus

from utils import load_cistopic_obj, save_cistopic_obj


log.basicConfig(level=log.INFO)
warnings.simplefilter(action="ignore")
sc.settings.set_figure_params(
    dpi=150, frameon=False, figsize=(10, 10), facecolor="white"
)


###########################################

infile_rna_sample_id_col = "sample_id_obs_column"
infile_rna_cell_type_col = "cell_type_obs_column"
tf_file = "/path/to/tf/list/utoronto_human_tfs_v_1.01.txt"
biomart_host = "http://www.ensembl.org"
path_bedToBigBed = "/path/to/dir/with/bedToBigBed"  # http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bedToBigBed
work_dir = Path("/path/to/work_dir")
tmp_dir = tempfile.mkdtemp()
spill_dir = tempfile.mkdtemp(dir="/path/to/spill_dir")
n_cores = 24
is_raw = True

###########################################

atac_path = work_dir / "scATAC"
scenicplus_path = work_dir / "scenicplus"
motifs_path = work_dir / "motifs"

infile_rna_h5ad = work_dir / "scRNA" / "anndata_metacells.h5ad"

log.info(f"ray tmp dir: {tmp_dir}")
log.info(f"ray spill dir: {spill_dir}")

# run Scenic+
if (scenicplus_path / "scplus_obj.pkl").is_file():
    log.info("object exists, loading scenic object...")
    with open(scenicplus_path / "scplus_obj.pkl", "rb") as f:
        scplus_obj = dill.load(f)
else:
    # create scenic+ object

    log.info("creating a new scenic object...")

    cistopic_obj = load_cistopic_obj(atac_path / "cistopic_obj_filt.pkl")

    log.info("load motif enrichment results...")

    with open(motifs_path / "menr.pkl", "rb") as f:
        menr = dill.load(f)

    log.info("load cell metadata from RNA object")

    adata = sc.read_h5ad(infile_rna_h5ad)
    cell_data = adata.obs

    if adata.X.min() < 0:
        if adata.raw:
            adata = adata.raw.to_adata()
        else:
            log.error("anndata seems to contain scaled counts")
            raise ValueError("anndata seems to contain scaled counts")

    if adata.X.max() < 100 or is_raw:
        if adata.raw:
            adata = adata.raw.to_adata()
        else:
            log.warning("anndata seems to be log-normalised")

    if infile_rna_sample_id_col:
        cell_data["sample_id"] = cell_data[infile_rna_sample_id_col].astype(str)

    def bc_transform(bc_gex):
        """transform GEX barcode from h5ad to ATAC barcode format"""
        return bc_gex

    log.info(f"GEX barcodes: {', '.join(bc_transform(x) for x in adata.obs_names[:5])}")
    log.info(f"ATAC barcodes: {', '.join(x for x in cistopic_obj.cell_data.index[:5])}")
    log.info(
        f"overlap: {len(set(bc_transform(x) for x in adata.obs_names) & set(cistopic_obj.cell_data.index))}"
    )

    log.info("create scenicplus object")

    scplus_obj = create_SCENICPLUS_object(
        GEX_anndata=adata,
        bc_transform_func=bc_transform,
        cisTopic_obj=cistopic_obj,
        menr=menr,
    )

    try:
        scplus_obj.X_EXP = np.array(scplus_obj.X_EXP.todense())
    except BaseException as e:
        scplus_obj.X_EXP = np.array(scplus_obj.X_EXP)
        log.warning(e)

    # only keep the first two columns of the PCA embedding in order to be able to visualize this in SCope
    try:
        scplus_obj.dr_cell["GEX_X_pca"] = scplus_obj.dr_cell["GEX_X_pca"].iloc[:, 0:2]
    except BaseException as e:
        log.warning(e)

    log.info("check biomart host...")

    if not biomart_host:
        ensembl_version_dict = {
            "105": "http://www.ensembl.org",
            "104": "http://may2021.archive.ensembl.org/",
            "103": "http://feb2021.archive.ensembl.org/",
            "102": "http://nov2020.archive.ensembl.org/",
            "101": "http://aug2020.archive.ensembl.org/",
            "100": "http://apr2020.archive.ensembl.org/",
            "99": "http://jan2020.archive.ensembl.org/",
            "98": "http://sep2019.archive.ensembl.org/",
            "97": "http://jul2019.archive.ensembl.org/",
            "96": "http://apr2019.archive.ensembl.org/",
            "95": "http://jan2019.archive.ensembl.org/",
            "94": "http://oct2018.archive.ensembl.org/",
            "93": "http://jul2018.archive.ensembl.org/",
            "92": "http://apr2018.archive.ensembl.org/",
            "91": "http://dec2017.archive.ensembl.org/",
            "90": "http://aug2017.archive.ensembl.org/",
            "89": "http://may2017.archive.ensembl.org/",
            "88": "http://mar2017.archive.ensembl.org/",
            "87": "http://dec2016.archive.ensembl.org/",
            "86": "http://oct2016.archive.ensembl.org/",
            "80": "http://may2015.archive.ensembl.org/",
            "77": "http://oct2014.archive.ensembl.org/",
            "75": "http://feb2014.archive.ensembl.org/",
            "54": "http://may2009.archive.ensembl.org/",
        }

        def test_ensembl_host(scplus_obj, host, species):
            dataset = pbm.Dataset(name=species + "_gene_ensembl", host=host)
            annot = dataset.query(
                attributes=[
                    "chromosome_name",
                    "transcription_start_site",
                    "strand",
                    "external_gene_name",
                    "transcript_biotype",
                ]
            )
            annot.columns = ["Chromosome", "Start", "Strand", "Gene", "Transcript_type"]
            annot["Chromosome"] = annot["Chromosome"].astype("str")
            filter = annot["Chromosome"].str.contains("CHR|GL|JH|MT")
            annot = annot[~filter]
            annot.columns = ["Chromosome", "Start", "Strand", "Gene", "Transcript_type"]
            gene_names_release = set(annot["Gene"].tolist())
            ov = len([x for x in scplus_obj.gene_names if x in gene_names_release])
            print(
                "Genes recovered: "
                + str(ov)
                + " out of "
                + str(len(scplus_obj.gene_names))
            )
            return ov

        n_overlap = {}
        for version in ensembl_version_dict.keys():
            print(f"host: {version}")
            try:
                n_overlap[version] = test_ensembl_host(
                    scplus_obj, ensembl_version_dict[version], "hsapiens"
                )
            except:
                print("Host not reachable")
        v = sorted(n_overlap.items(), key=lambda item: item[1], reverse=True)[0][0]
        print(
            f"version: {v} has the largest overlap, use {ensembl_version_dict[v]} as biomart host"
        )

        biomart_host = ensembl_version_dict[v]


# run Scenic+ analysis step

log.info("run Scenic+")

try:
    scplus_cell_type_col = f"GEX_{infile_rna_cell_type_col}"
    calculate_TF_eGRN_correlation = True
    if (scplus_obj.metadata_cell[scplus_cell_type_col].value_counts() <= 5).any():
        log.warning(
            f"cell types with <=5 cells in scenicplus object '{scplus_cell_type_col}' column, "
            "disabling 'calculate_TF_eGRN_correlation'"
        )
        calculate_TF_eGRN_correlation = False

    os.environ["TMPDIR"] = f"{tmp_dir}"
    gc.set_threshold(300, 5, 5)

    run_scenicplus(
        scplus_obj=scplus_obj,
        variable=[scplus_cell_type_col],
        species="hsapiens",
        assembly="hg38",
        tf_file=tf_file,
        save_path=str(scenicplus_path) + "/",
        biomart_host=biomart_host,
        upstream=[1000, 150000],
        downstream=[1000, 150000],
        calculate_TF_eGRN_correlation=calculate_TF_eGRN_correlation,
        calculate_DEGs_DARs=False,
        export_to_loom_file=False,
        export_to_UCSC_file=False,
        path_bedToBigBed=path_bedToBigBed,
        n_cpu=n_cores,
        _temp_dir=tmp_dir,
        _system_config={
            "max_io_workers": 4,  # More IO workers for parallelism.
            "object_spilling_config": json.dumps(
                {
                    "type": "filesystem",
                    "params": {
                        "directory_path": [
                            spill_dir,
                        ]
                    },
                }
            ),
        },
    )
except Exception as e:
    # in case of failure, still save the object
    with open(scenicplus_path / "scplus_obj.pkl", "wb") as f:
        dill.dump(scplus_obj, f, protocol=-1)
    raise e

# save

log.info(f"save Scenic+ object to {str(scenicplus_path / 'scplus_obj.pkl')}")

with open(scenicplus_path / "scplus_obj.pkl", "wb") as f:
    dill.dump(scplus_obj, f, protocol=-1)

log.info("all done.")
