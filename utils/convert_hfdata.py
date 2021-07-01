

import json
import pickle as pkl
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


hf_icd = ['490', '491', '491.0', '491.1', '491.2', '491.20', '491.21', '491.22',
        '491.8', '491.9', '492', '492.0', '492.8', '494', '494.0', '494.1', '496']


def load_code2idx(code2idx_path: str) -> dict:
    with open(code2idx_path, "rb") as fin:
        code2idx = pkl.load(fin)
    return code2idx


def map_icd2cui(snomedct_path: str, icd2snomed_1to1_path: str, icd2snomed_1toM_path: str) -> dict:
    """
    convert icd-9 codes to umls_cui via snomed_cid

    `return`:
        icd2cui: dict: {"icd_code (str)": umls_cui (str)}
        icd_list: list of all icd codes in the icd2cui dictionary
    """
    # load snomed and icd2snomed
    snomed = pd.read_table(snomedct_path, sep="|")
    snomed2cui = snomed[["SNOMED_CID", "UMLS_CUI"]]
    icd2snomed_1to1 = pd.read_table(icd2snomed_1to1_path)
    icd2snomed_1toM = pd.read_table(icd2snomed_1toM_path)
    # convert the dataframe into dictionary
    snomed2cui_dict = snomed2cui.set_index("SNOMED_CID").T.to_dict("records")[0]  # dict: {"snomed_cid (int)": umls_cui (str)}
    # map cui to icd2snomed via snomed
    icd2snomed_1to1["UMLS_CUI"] = icd2snomed_1to1["SNOMED_CID"].map(snomed2cui_dict)
    icd2snomed_1toM["UMLS_CUI"] = icd2snomed_1toM["SNOMED_CID"].map(snomed2cui_dict)
    # drop all rows that have any NaN values
    icd2snomed_1to1 = icd2snomed_1to1.dropna(axis=0, how="any")
    icd2snomed_1toM = icd2snomed_1toM.dropna(axis=0, how="any")
    # extract icd and cui
    icd_cui_1to1 = icd2snomed_1to1[["ICD_CODE", "UMLS_CUI"]]
    icd_cui_1toM = icd2snomed_1toM[["ICD_CODE", "UMLS_CUI"]]
    # drop duplicates in icd codes
    icd_cui_1toM = icd_cui_1toM.drop_duplicates(subset=["ICD_CODE"], keep="first")
    # convert the dataframe into dictionary
    icd2cui_1to1 = icd_cui_1to1.set_index("ICD_CODE").T.to_dict("records")[0]
    icd2cui_1toM = icd_cui_1toM.set_index("ICD_CODE").T.to_dict("records")[0]
    icd2cui = {}
    icd2cui.update(icd2cui_1to1)
    icd2cui.update(icd2cui_1toM)
    # make the list of all icd codes in the dictionary
    icd_list = list(icd2cui.keys())
    cui_list = list(icd2cui.values())
    return icd2cui, icd_list, cui_list


def create_output_dict(row, medical_record, hf_label, icd_list, idx2code, icd2cui, time_dist):
    """
    convert each line of the hf data into a dictionary

    `param`:
        medical_record: the medical history of one patient
        hf_label: int (1/0) that indicates if this patient has a heart disease later
        idx2code: dict: {"idx (int)": icd_code (str)}
        icd2cui: dict: {"icd_code (str)": umls_cui (str)}
    `return`:
        output_dict: {}
    """
    output_dict = {}
    output_dict["id"] = 0
    output_dict["medical_records"] = 0
    output_dict["heart_diseases"] = 0
    record_icd = []
    record_cui = []
    hf_cui = []

    # if there're over 50 visits, select the latest 50 ones
    if len(medical_record) > 50:
        medical_record = medical_record[-50:]
        time_dist = time_dist[-50:]

    for visit in medical_record:
        visit_icd = []
        visit_cui = []
        for idx in visit:
            icd = idx2code[idx]
            visit_icd.append(icd)
            if icd in icd_list:
                cui = icd2cui[icd]
                visit_cui.append(cui)
            # else: record_cui = []
        record_icd.append(visit_icd)
        record_cui.append(visit_cui)
    for icd in hf_icd:
        if icd in icd_list:
            cui = icd2cui[icd]
        hf_cui.append(cui)
    # hf_label is NumPy.int64 and json does not recognize NumPy data types
    hf_label = int(hf_label)
    output_dict["id"] = row
    output_dict["medical_records"] = {"record_icd": record_icd, "time_distance": time_dist, "record_cui": record_cui}
    output_dict["heart_diseases"] = {"hf_icd": hf_icd, "hf_cui": hf_cui, "hf_label": hf_label}
    return output_dict


def convert_to_cui(pickle_file, output_file, code2idx_path, snomedct_path, icd2snomed_1to1_path, icd2snomed_1toM_path):
    """
    Read input data in pickle format, convert them into the dictionary
    and write the converted data in jsonl format
    """
    print(f'converting {pickle_file} to jsonl file')
    # prepare mapping dictionaries
    code2idx = load_code2idx(code2idx_path) # dict: {"icd_code (str)": idx (int)}
    idx2code = dict([val, key] for key, val in code2idx.items())
    icd2cui, icd_list, cui_list = map_icd2cui(snomedct_path, icd2snomed_1to1_path, icd2snomed_1toM_path)

    # read the input file in pickle format
    with open(pickle_file, "rb") as fin:
        hf = pkl.load(fin)
    medical_records = hf[0]
    hf_labels = hf[1]
    time_dist = hf[2]

    nrow = len(hf_labels)

    # write the output file in jsonl format line by line
    with open(output_file, "w") as fout:
        for row in tqdm(range(nrow)):
            output_dict = create_output_dict(row, medical_records[row], hf_labels[row], icd_list, idx2code, icd2cui, time_dist[row])
            fout.write(json.dumps(output_dict))
            fout.write("\n")
    print(f'converted data saved to {output_file}')
    print()


if __name__ == "__main__":
    convert_to_cui('../data/hfdata/hf_validation_new.pickle','../data/hfdata/converted/dev.jsonl','../data/hfdata/hf_code2idx_new.pickle',
                   '../data/semmed/SNOMEDCT_CORE_SUBSET_202002.txt','../data/semmed/ICD9CM_SNOMED_MAP_1TO1_201912.txt',
                   '../data/semmed/ICD9CM_SNOMED_MAP_1TOM_201912.txt')
    convert_to_cui('../data/hfdata/hf_training_new.pickle', '../data/hfdata/converted/train.jsonl',
                   '../data/hfdata/hf_code2idx_new.pickle',
                   '../data/semmed/SNOMEDCT_CORE_SUBSET_202002.txt', '../data/semmed/ICD9CM_SNOMED_MAP_1TO1_201912.txt',
                   '../data/semmed/ICD9CM_SNOMED_MAP_1TOM_201912.txt')
    convert_to_cui('../data/hfdata/hf_testing_new.pickle', '../data/hfdata/converted/test.jsonl',
                   '../data/hfdata/hf_code2idx_new.pickle',
                   '../data/semmed/SNOMEDCT_CORE_SUBSET_202002.txt', '../data/semmed/ICD9CM_SNOMED_MAP_1TO1_201912.txt',
                   '../data/semmed/ICD9CM_SNOMED_MAP_1TOM_201912.txt')
