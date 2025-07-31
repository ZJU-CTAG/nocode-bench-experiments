import argparse
import os
import shutil
import zipfile
from huggingface_hub import HfApi, hf_hub_download, login


def download_from_hf(repo_id, repo_type, target_folder=None, local_dir="./downloads", unzip=True):
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    os.makedirs(local_dir, exist_ok=True)

    for file_path in repo_files:
        if target_folder and not file_path.startswith(target_folder):
            continue

        local_path = os.path.join(local_dir, file_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=file_path,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False
        )

        if unzip and downloaded_file.endswith(".zip"):
            with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
                extract_path = os.path.splitext(downloaded_file)[0]
                zip_ref.extractall(extract_path)
            os.remove(downloaded_file)


def main(args):
    # token = os.environ.get("HF_TOKEN")
    # if not token:
    #     return
    # login(token)

    if args.logs:
        download_from_hf(
            repo_id="NoCode-bench/Logs",
            repo_type="dataset",
            target_folder=args.target_folder,
            local_dir=args.local_dir,
            unzip=not args.no_unzip
        )

    if args.trajs:
        download_from_hf(
            repo_id="NoCode-bench/Trajs",
            repo_type="dataset",
            target_folder=args.target_folder,
            local_dir=args.local_dir,
            unzip=not args.no_unzip
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", action="store_true", help="Download from Logs repo")
    parser.add_argument("--trajs", action="store_true", help="Download from Trajs repo")
    parser.add_argument("--target_folder", type=str, help="Target folder inside repo to download", default=None)
    parser.add_argument("--local_dir", type=str, help="Local directory to save files", default="./downloads")
    parser.add_argument("--no_unzip", action="store_true", help="Do not unzip .zip files after downloading")
    main(parser.parse_args())
