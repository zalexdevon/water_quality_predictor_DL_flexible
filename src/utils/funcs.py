from Mylib import myfuncs
import time
import os
import re
from sklearn.pipeline import Pipeline
from src.utils import classes
import pandas as pd
import plotly.express as px


def get_train_val_scoring_to_plot(train_scoring, val_scoring, scoring):
    if scoring == "accuracy":
        train_scoring, val_scoring = train_scoring * 100, val_scoring * 100

    return train_scoring, val_scoring


def train_and_save_models(
    model_training_path,
    model_name,
    model_indices,
    train_feature_data,
    train_target_data,
    val_feature_data,
    val_target_data,
    scoring,
    plot_dir,
    num_models,
):
    print(
        f"\n========Bắt đầu train {num_models} models ở chế độ bình thường !!!!!!================\n"
    )
    start_time = time.time()  # Bắt đầu tính thời gian train model
    for model_index in model_indices:
        # Load model để train
        model = myfuncs.load_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl")
        )

        print(f"Bắt đầu train model {model_name} - {model_index}")
        model.fit(train_feature_data, train_target_data)
        print(f"Kết thúc train model {model_name} - {model_index}")

        train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            train_feature_data,
            train_target_data,
            scoring,
        )
        val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            val_feature_data,
            val_target_data,
            scoring,
        )

        # In kết quả
        print("Kết quả của model")
        print(
            f"Model index {model_name} - {model_index}\n -> Train {scoring}: {train_scoring}, Val {scoring}: {val_scoring}\n"
        )

        # Lưu model sau khi trained
        myfuncs.save_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl"), model
        )

        # Lưu dữ liệu để vẽ biểu đồ
        model_name_in_plot = f"{model_name}_{model_index}"

        train_scoring, val_scoring = get_train_val_scoring_to_plot(
            train_scoring, val_scoring, scoring
        )

        myfuncs.save_python_object(
            os.path.join(plot_dir, f"{model_name_in_plot}.pkl"),
            (model_name_in_plot, train_scoring, val_scoring),
        )

    all_model_end_time = time.time()  # Kết thúc tính thời gian train model
    true_all_models_train_time = (all_model_end_time - start_time) / 60

    print(f"Thời gian chạy tất cả: {true_all_models_train_time} (mins)")

    print(
        f"\n========Kết thúc train {num_models} models ở chế độ bình thường !!!!!!================\n"
    )


def get_batch_size_from_model_training_name(name):
    pattern = r"batch_(\d+)"
    return int(re.findall(pattern, name)[0])


def print_text(text):
    print(f"Demo cho utils nè mấy ní ơi !!!!!! {text}")


def plot_from_components(components, max_val_value, target_val_value, dtick_y_value):
    # Vẽ biểu đồ từ các components
    model_names = [item[0] for item in components]
    train_scores = [item[1] for item in components]
    val_scores = [item[2] for item in components]

    for i in range(len(train_scores)):
        if train_scores[i] > max_val_value:
            train_scores[i] = max_val_value

        if val_scores[i] > max_val_value:
            val_scores[i] = max_val_value

    # Vẽ biểu đồ
    df = pd.DataFrame(
        {
            "x": model_names,
            "train": train_scores,
            "val": val_scores,
        }
    )

    df_long = df.melt(
        id_vars=["x"],
        value_vars=["train", "val"],
        var_name="Category",
        value_name="y",
    )

    fig = px.line(
        df_long,
        x="x",
        y="y",
        color="Category",
        markers=True,
        color_discrete_map={
            "train": "gray",
            "val": "blue",
        },
        hover_data={"x": False, "y": True, "Category": False},
    )

    fig.add_hline(
        y=max_val_value,
        line_dash="solid",
        line_color="black",
        line_width=2,
    )

    fig.add_hline(
        y=target_val_value,
        line_dash="dash",
        line_color="green",
        line_width=2,
    )

    fig.update_layout(
        autosize=False,
        width=100 * (len(model_names) + 2) + 30,
        height=400,
        margin=dict(l=30, r=10, t=10, b=0),
        xaxis=dict(
            title="",
            range=[
                0,
                len(model_names) + 2,
            ],
            tickmode="linear",
        ),
        yaxis=dict(
            title="",
            range=[0, max_val_value + dtick_y_value],
            dtick=dtick_y_value,
        ),
        showlegend=False,
    )

    return fig


def sort_components_by_target_val_value(components, target_val_value):
    model_names = pd.Series([item[0] for item in components])
    train_scores = pd.Series([item[1] for item in components])
    val_scores = pd.Series([item[2] for item in components])

    # Tiến hành cho các chỉ số mà ưu tiên lớn hơn, vd: accuracy, ...
    ## Lọc ra 2 phần both lớn hơn target_val_value và ngược lại
    mask = (train_scores > target_val_value) & (val_scores > target_val_value)
    train_ge_target_val_value = train_scores[mask]
    train_left = train_scores[~train_scores.index.isin(train_ge_target_val_value.index)]

    val_ge_target_val_value = val_scores[mask]
    val_left = val_scores[~val_scores.index.isin(val_ge_target_val_value.index)]

    model_ge_target_val_value = model_names[mask]
    model_left = model_names[~model_names.index.isin(model_ge_target_val_value.index)]

    ## Sắp xếp phần both lớn hơn
    val_ge_target_val_value = val_ge_target_val_value.sort_values(ascending=False)
    train_ge_target_val_value = train_ge_target_val_value[val_ge_target_val_value.index]
    model_ge_target_val_value = model_ge_target_val_value[val_ge_target_val_value.index]

    ## Sắp xếp phần còn lại
    val_left = val_left.sort_values(ascending=False)
    train_left = train_left[val_left.index]
    model_left = model_left[val_left.index]

    # Gộp 2 phần lại
    train_result = pd.concat([train_ge_target_val_value, train_left], axis=0).tolist()
    val_result = pd.concat([val_ge_target_val_value, val_left], axis=0).tolist()
    model_result = pd.concat([model_ge_target_val_value, model_left], axis=0).tolist()

    return zip(model_result, train_result, val_result)


def plot_train_val_scoring_from_trained_models(
    max_val_value, target_val_value, dtick_y_value
):
    plot_dir = "artifacts/plot/components"

    # Get các components
    component_paths = os.listdir(plot_dir)
    components = [
        myfuncs.load_python_object(os.path.join(plot_dir, item))
        for item in component_paths
    ]

    # Sắp xếp các components ưu tiên các cặp (train_scoring, val_scoring) cùng lớn hơn target_val_value trước
    components = sort_components_by_target_val_value(components, target_val_value)

    # Vẽ biểu đồ
    return plot_from_components(
        components, max_val_value, target_val_value, dtick_y_value
    )
