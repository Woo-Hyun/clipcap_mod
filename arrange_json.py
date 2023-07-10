import json

def pretty_json(json_file, out_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(out_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)

json_file = "../drama_data/integrated_output_clipcap.json"  # 여기에 json 파일 이름을 입력하세요.
out_file = "../drama_data/integrated_output_clipcap_arrage.json"
pretty_json(json_file, out_file)
