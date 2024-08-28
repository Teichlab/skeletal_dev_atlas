#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import tempfile
import logging as log
import warnings
from pathlib import Path
from pycisTopic.cistopic_class import *
from pycisTopic.lda_models import *
from pycisTopic.clust_vis import *
from pycisTopic.topic_binarization import *
from pycisTopic.diff_features import *

from utils import load_cistopic_obj, save_cistopic_obj

log.basicConfig(level=log.INFO)
warnings.simplefilter(action="ignore", category=FutureWarning)

###########################################

outfile_models = "/path/to/scATAC/models/models_LDA.pkl"
work_dir = Path("/path/to/work_dir")
n_cores = 4
topic_modeling_n_iter = 500

tmp_dir = tempfile.mkdtemp()

###########################################

atac_path = work_dir / "scATAC"
qc_path = atac_path / "quality_control"
candidate_enhancer_path = atac_path / "candidate_enhancers"


log.info("create directories")

if not os.path.exists(os.path.join(work_dir, "scATAC")):
    os.makedirs(os.path.join(work_dir, "scATAC"))


log.info("load cistopic object")

cistopic_obj = load_cistopic_obj(atac_path / "cistopic_obj_filt.pkl")


log.info("run models")

models = run_cgs_models(
    cistopic_obj,
    n_topics=[2, 4, 10, 16, 32, 48],
    n_cpu=n_cores,
    n_iter=topic_modeling_n_iter,
    random_state=555,
    alpha=50,
    alpha_by_topic=True,
    eta=0.1,
    eta_by_topic=False,
    save_path=None,
    _temp_dir=tmp_dir,
)


# Save results

log.info("save results")

if not os.path.exists(os.path.join(work_dir, "scATAC/models")):
    os.makedirs(os.path.join(work_dir, "scATAC/models"))

with open(outfile_models, "wb") as f:
    pickle.dump(models, f)
