import json


def overwrite_dict(org_dict, sub_dict):
    for sub_key, sub_value in sub_dict.items():
        if sub_key in org_dict:
            if isinstance(sub_value, dict):
                overwrite_dict(org_dict[sub_key], sub_value)
            else:
                org_dict[sub_key] = sub_value
        else:
            org_dict[sub_key] = sub_value


def overwrite_config(config, json_str):
    overwrite_dict(config, json.loads(json_str))
