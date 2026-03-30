import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_data(
    repo_id, 
    token, 
    base_dir="./data", 
    private=True
):
    """
    repo_id: 你的HF用户名/仓库名 (例如: maoy0027/pdf-mineru-dataset)
    token: 你的 Write 权限 Token
    base_dir: 你要上传的数据根目录
    private: 是否创建为私有仓库
    """
    api = HfApi(token=token)
    
    try:
        url = create_repo(
            repo_id=repo_id, 
            token=token, 
            private=private, 
            repo_type="dataset", 
            exist_ok=True
        )
        print(f"✅ Repository ready: {url}")
    except Exception as e:
        print(f"⚠️ Repo creation warning: {e}")

    print("🚀 Uploading Raw PDFs...")
    api.upload_folder(
        folder_path=os.path.join(base_dir, "raw_pdfs"),
        path_in_repo="data/raw_pdfs",
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=[".DS_Store", "*.tmp"],
    )

    print("🚀 Uploading MinerU Outputs (This may take a while)...")
    api.upload_folder(
        folder_path=os.path.join(base_dir, "mineru_output"),
        path_in_repo="data/mineru_output",
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=[".DS_Store", "*.tmp"],
    )
    
    # 4. 上传其他重要文件 (如 metadata)
    # 如果你有 triplets.jsonl 等文件，也可以顺便传上去
    for filename in ["triplets_v2.jsonl", "paper_metadata.json"]:
        file_path = os.path.join(base_dir, filename) # 修改路径
        # 如果文件在项目根目录，修改 path
        if os.path.exists(filename):
             print(f"🚀 Uploading {filename}...")
             api.upload_file(
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset"
             )

    print("🎉 All Uploads Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="Format: username/dataset_name")
    parser.add_argument("--token", type=str, help="HF Access Token")
    args = parser.parse_args()
    
    token = os.environ.get("HF_TOKEN") or args.token
    if not token:
        raise ValueError("Please provide a token via --token or HF_TOKEN env var")

    upload_data(args.repo, token)