import json
import os

def generate_configurations(base_dir, aggregation_methods, alphas, zip_percents, scale_factors):
    os.makedirs(base_dir, exist_ok=True)

    for method in aggregation_methods:
        method_dir = os.path.join(base_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        for alpha in alphas:
            for zip_percent in zip_percents:
                for scale_down in scale_factors:
                    config_name = f"alpha_{alpha}_zip_{zip_percent}_scale_{scale_down}.json"
                    config_path = os.path.join(method_dir, config_name)

                    config = [{
                        "num_rounds": 2,
                        "num_clients": 2,
                        "num_classes_per_client": 3,
                        "batch_size": 64,
                        "iid_setting": False,
                        "special_distribution": False,
                        "dirichlet": True,
                        "class_spesific": False,
                        "aggregation_method": method,
                        "top_k": 3,
                        "scale_down_factor": scale_down if method != "fedavg" else None,
                        "zip_percent": zip_percent if method != "fedavg" else None,
                        "alpha": alpha
                    }]

                    with open(config_path, "w") as file:
                        json.dump(config, file, indent=4)


generate_configurations(
    base_dir="configurations",
    aggregation_methods=[ "lm_mask","gradcam", "fedavg"],
    alphas=[0.3],
    zip_percents=[0.4, 0.6],
    scale_factors=[0.3, 0.6]
)