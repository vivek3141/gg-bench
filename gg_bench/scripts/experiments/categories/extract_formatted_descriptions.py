import os
import json

def compile_descriptions(
    valid_envs_path: str,
    descriptions_dir: str, 
    output_json_path: str
):
    """
    1. Reads the valid_envs.json which is in the following format: [["272, 250000], ["154", 750000], ...].
    2. extracts the first elements in each sub list so: 272, 154
    3. reads each corresponding .txt file corresponding to those indices 
    4. compiles into a JSON object file: 
    Args:
        valid_envs_path (str): _description_
        descriptions_dir (str): _description_
        output_json_path (str): _description_
    """
    
    with open(valid_envs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    valid_indices = [pair[0] for pair in data] # ["272", "154"] strings formatted
    
    # for each index in valid incides, read .txt file correspondingly
    text_contents = []
    for idx in valid_indices:
        txt_filename = f"{idx}.txt"
        txt_path = os.path.join(descriptions_dir, txt_filename)
        # Check if the .txt file exists
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as txt_file:
                content = txt_file.read().strip()
                text_contents.append(content)
        else:
            print(f"Warning: {txt_path} does not exist, skipping.")
            
    
    # build the json structure properly
    output_data = {
        "goal": "I would like to cluster them based on topic; each cluster should have a description of 'has a topic of <something>'", 
        "texts": text_contents
    }
    
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(output_data, out_f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    valid_envs_path = "gg_bench/data/splits/valid_envs.json"
    descriptions_dir = "gg_bench/data/descriptions"
    output_json_path = "compiled_descriptions.json"

    compile_descriptions(valid_envs_path, descriptions_dir, output_json_path)
    print(f"Done! Wrote compiled JSON to {output_json_path}")
    