import pandas as pd
from Mylib import myfuncs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class BeforeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        # Xóa các cột không cần thiết
        df = df.drop(
            columns=[
                "taken_time",
            ]
        )

        #  Đổi tên cột
        rename_dict = {
            "pH": "pH_num",
            "Iron": "Iron_num",
            "Nitrate": "Nitrate_num",
            "Chloride": "Chloride_num",
            "Lead": "Lead_num",
            "Zinc": "Zinc_num",
            "Color": "Color_ord",
            "Turbidity": "Turbidity_num",
            "Fluoride": "Fluoride_num",
            "Copper": "Copper_num",
            "Odor": "Odor_num",
            "Sulfate": "Sulfate_num",
            "Conductivity": "Conductivity_num",
            "Chlorine": "Chlorine_num",
            "Manganese": "Manganese_num",
            "Total Dissolved Solids": "Total_Dissolved_Solids_num",
            "Source": "Source_nom",
            "Water Temperature": "Water_Temperature_num",
            "Air Temperature": "Air_Temperature_num",
            "Target": "Target_target",
        }

        df = df.rename(columns=rename_dict)

        # Sắp xếp các cột theo đúng thứ tự
        (
            numeric_cols,
            numericCat_cols,
            cat_cols,
            binary_cols,
            nominal_cols,
            ordinal_cols,
            target_col,
        ) = myfuncs.get_different_types_cols_from_df_4(df)

        df = df[
            numeric_cols
            + numericCat_cols
            + binary_cols
            + nominal_cols
            + ordinal_cols
            + [target_col]
        ]

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        df = X

        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        self.handler = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="mean"), numeric_cols),
                (
                    "numCat",
                    SimpleImputer(strategy="most_frequent"),
                    numericCat_cols,
                ),
                ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
                ("target", SimpleImputer(strategy="most_frequent"), [target_col]),
            ]
        )
        self.handler.fit(df)

    def transform(self, X, y=None):
        df = X

        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        df = self.handler.transform(df)
        self.cols = numeric_cols + numericCat_cols + cat_cols + [target_col]
        df = pd.DataFrame(df, columns=self.cols)

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class AfterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        self.cols = df.columns.tolist()

        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        # Chuyển đổi về đúng kiểu dữ liệu
        df[numeric_cols] = df[numeric_cols].astype("float32")
        df[numericCat_cols] = df[numericCat_cols].astype("float32")
        df[cat_cols] = df[cat_cols].astype("category")
        df[target_col] = df[target_col].astype("category")

        # Loại bỏ duplicates
        df = df.drop_duplicates().reset_index(drop=True)

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols
