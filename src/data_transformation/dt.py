from Mylib import myfuncs, stringToObjectConverter, myclasses, tf_myfuncs
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
)
import tensorflow as tf


def load_data_for_data_transformation(data_correction_path):
    # Load df train đã corrected
    df_train = myfuncs.load_python_object(
        os.path.join(data_correction_path, "data.pkl")
    )

    # Load dict để biến đổi các biến ordinal
    feature_ordinal_dict = myfuncs.load_python_object(
        os.path.join(
            data_correction_path,
            "feature_ordinal_dict.pkl",
        )
    )

    # Load transformer của data correction để sau này transform val_data
    correction_transformer = myfuncs.load_python_object(
        os.path.join(data_correction_path, "transformer.pkl")
    )

    # Load val data đã corrected
    val_data_path = "artifacts/data_ingestion/val_data.pkl"
    df_val = myfuncs.load_python_object(val_data_path)

    # Các cột feature và cột target
    feature_cols, target_col = myfuncs.get_feature_cols_and_target_col_from_df_27(
        df_train
    )

    return (
        df_train,
        feature_ordinal_dict,
        correction_transformer,
        df_val,
        feature_cols,
        target_col,
    )


def create_data_transformation_transformer(
    list_after_feature_transformer, feature_ordinal_dict, feature_cols, target_col
):
    after_feature_pipeline = myfuncs.convert_list_estimator_into_pipeline_59(
        list_after_feature_transformer
    )

    feature_pipeline = Pipeline(
        steps=[
            (
                "during",
                myclasses.DuringFeatureTransformer(feature_ordinal_dict),
            ),
            ("after", after_feature_pipeline),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("feature", feature_pipeline, feature_cols),
            ("target", OrdinalEncoder(), [target_col]),
        ]
    )

    transformation_transformer = myclasses.NamedColumnTransformer(column_transformer)

    return transformation_transformer


def do_transform_data_in_data_transformation(
    transformation_transformer,
    df_train,
    df_val,
    target_col,
    correction_transformer,
    batch_size,
):
    df_train_transformed = transformation_transformer.fit_transform(df_train)
    df_train_feature = df_train_transformed.drop(columns=[target_col]).astype("float32")
    df_train_target = df_train_transformed[target_col].astype("int8")

    df_val_corrected = correction_transformer.transform(df_val)
    df_val_transformed = transformation_transformer.transform(df_val_corrected)
    df_val_feature = df_val_transformed.drop(columns=[target_col]).astype("float32")
    df_val_target = df_val_transformed[target_col].astype("int8")

    # Get shape của train features
    train_feature_shape = df_train_feature.shape

    # Convert thành dataset
    train_ds = tf.data.Dataset.from_tensor_slices((df_train_feature, df_train_target))
    train_ds = train_ds.batch(batch_size)
    train_ds = tf_myfuncs.cache_prefetch_tfdataset_2(train_ds)

    val_ds = tf.data.Dataset.from_tensor_slices((df_val_feature, df_val_target))
    val_ds = val_ds.batch(batch_size)
    val_ds = tf_myfuncs.cache_prefetch_tfdataset_2(val_ds)

    return train_ds, val_ds, train_feature_shape


def save_data_for_data_transformation(
    data_transformation_path,
    transformation_transformer,
    batch_size,
    train_ds,
    val_ds,
    train_feature_shape,
):
    myfuncs.save_python_object(
        f"{data_transformation_path}/transformer.pkl", transformation_transformer
    )
    myfuncs.save_python_object(f"{data_transformation_path}/batch_size.pkl", batch_size)
    train_ds.save(f"{data_transformation_path}/train_ds.pkl")
    val_ds.save(f"{data_transformation_path}/val_ds.pkl")

    # Save số lượng features của tập training
    num_features = train_feature_shape[1]
    myfuncs.save_python_object(
        f"{data_transformation_path}/num_features.pkl", num_features
    )


def create_weight_data_transformation_transformer(
    weights,
    list_after_feature_transformer,
    feature_ordinal_dict,
    feature_cols,
    target_col,
):
    after_feature_pipeline = myfuncs.convert_list_estimator_into_pipeline_59(
        list_after_feature_transformer
    )

    feature_pipeline = Pipeline(
        steps=[
            (
                "during",
                myclasses.DuringFeatureTransformer(feature_ordinal_dict),
            ),
            ("after", after_feature_pipeline),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("feature", feature_pipeline, feature_cols),
            ("target", OrdinalEncoder(), [target_col]),
        ]
    )

    column_transformer = Pipeline(
        steps=[
            ("1", column_transformer),
            (
                "2",
                myclasses.MultiplyWeightsTransformer(weights),
            ),  # Transformer cho weight
        ]
    )

    transformation_transformer = myclasses.NamedColumnTransformer(column_transformer)

    return transformation_transformer
