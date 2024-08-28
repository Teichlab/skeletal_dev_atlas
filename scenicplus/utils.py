import pickle
from pathlib import Path
import scipy as sp
import pandas as pd
from pycisTopic.cistopic_class import CistopicObject


def get_col_types(df):
    tolerance = round(df.shape[0] * 0.8)

    column_types = []
    static_vals = []
    for name, val in df.items():
        if val.value_counts()[0] > tolerance:
            column_types.append(val.value_counts().index[0])
            static_vals.append(val.value_counts().index[0])
        elif val.str.match("^[ACGT]+$").sum() > tolerance:
            column_types.append("barcode")
        elif val.str.match(".*cellranger").sum() > tolerance:
            column_types.append("cellranger_id")
        else:
            column_types.append("sample_id")

    return dict(
        dataframe=df.rename(columns={k: column_types[k] for k, v in df.items()}),
        static_cols=static_vals,
    )


def get_split_df(barcodes, split_pattern="-|___", cols_only=False):
    bc_df = pd.DataFrame({"barcode": barcodes})
    if cols_only:
        try:
            bc_df = bc_df.sample(500)
        except:
            pass
    bc_df = bc_df["barcode"].str.split(split_pattern, expand=True)

    res = get_col_types(bc_df)
    res_main = res["dataframe"] if not cols_only else res["dataframe"].columns

    return dict(res=res_main, static_cols=res["static_cols"])


def match_barcodes_to(from_bc, to_bc):
    to_parts = get_split_df(to_bc, cols_only=True, split_pattern="(-|___)")
    from_parts = get_split_df(from_bc)

    assembled_bc = ""
    for col in to_parts["res"]:
        if col not in to_parts["static_cols"]:
            try:
                assembled_bc += from_parts["res"][col]
            except KeyError as e:
                raise KeyError(f"{col} not found in 'from_bc': {e}")
        else:
            assembled_bc += col

    return assembled_bc.tolist()


def save_cistopic_obj(cistopic_obj, path):
    """
    save pycistopic object efficiently
    """
    save_npz = Path(path).with_suffix(".npz")
    save_pkl = Path(path).with_suffix(".pkl")

    print(f"save fragment matrix to: {save_npz}")
    sp.sparse.save_npz(save_npz, cistopic_obj.fragment_matrix)

    print(f"save object without matrices to: {save_pkl}")
    cistopic_obj_small = CistopicObject(
        None,
        None,
        cistopic_obj.cell_names,
        cistopic_obj.region_names,
        cistopic_obj.cell_data,
        cistopic_obj.region_data,
        cistopic_obj.path_to_fragments,
        cistopic_obj.project,
    )
    cistopic_obj_small.selected_model = cistopic_obj.selected_model
    cistopic_obj_small.projections = cistopic_obj.projections
    with open(save_pkl, "wb") as f:
        pickle.dump(cistopic_obj_small, f)

    print("all done.")


def load_cistopic_obj(path):
    """
    load pycistopic object efficiently
    """
    load_npz = Path(path).with_suffix(".npz")
    load_pkl = Path(path).with_suffix(".pkl")

    print(f"load object without matrices from: {load_pkl}")
    with open(load_pkl, "rb") as f:
        cistopic_obj_new = pickle.load(f)

    print(f"load fragment matrix from: {load_npz}")
    cistopic_obj_new.fragment_matrix = sp.sparse.load_npz(load_npz)

    print("restore binary matrix...")
    cistopic_obj_new.binary_matrix = (cistopic_obj_new.fragment_matrix > 0).astype(int)

    return cistopic_obj_new
