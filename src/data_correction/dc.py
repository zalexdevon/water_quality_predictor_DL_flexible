from Mylib import myfuncs
import os
from src.utils import funcs


def save_data_for_data_correction(
    data_correction_path, transformer, df_train_transformed, feature_ordinal_dict
):
    myfuncs.save_python_object(
        os.path.join(data_correction_path, "data.pkl"), df_train_transformed
    )
    myfuncs.save_python_object(
        os.path.join(data_correction_path, "feature_ordinal_dict.pkl"),
        feature_ordinal_dict,
    )
    myfuncs.save_python_object(
        os.path.join(data_correction_path, "transformer.pkl"), transformer
    )


def test_utils_run_on_python(text):
    return funcs.print_text(text)
