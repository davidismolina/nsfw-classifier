import os
import glob
import requests


txt_folder = "raw_data"

base_output_dir = "image_data"

os.makedirs(base_output_dir, exist_ok=True)

for txt_path in glob.glob(os.path.join(txt_folder, "*", "*.txt")):

    category = os.path.splitext(os.path.basename(txt_path))[0]
    output_dir = os.path.join(base_output_dir, category)
    os.makedirs(output_dir, exist_ok=True)

    with open(txt_path, 'r') as f:
        urls = [line.strip() for line in f]
    # print(urls[:6])
    print(f"Category: '{category}' {len(urls)}: Images")



    for idx, url in enumerate(urls, start=1):
        print(f"URL: {url}")
        try:
            # print(url)
            resp = requests.get(url, stream=True, timeout=10)
            resp.raise_for_status()
            filename = url.split("/")[-1].split("?")[0] or f"{category}_{idx}.jpg"
            
            print(filename)

            save_path = os.path.join(output_dir, filename)
            if os.path.exists(save_path):
                continue
            with open(save_path, 'wb') as out_file:
                for chunk in resp.iter_content(1024):
                    if chunk:
                        out_file.write(chunk)
                
            print(f"    [{category}] ({idx}/{len(urls)}) Saved: {filename}")
        except Exception as e:
            # print(f"    [{category}] ({idx}/{len(urls)}) Failed: {url} due to: {e}")
            print(f"    [{category}] ({idx}/{len(urls)}) Failed: {filename}")

    print("Done.")