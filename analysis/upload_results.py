import argparse
import os
import zipfile
import tempfile
from huggingface_hub import HfApi, login
from pathlib import Path


def zip_directory(directory, output_filename):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        base_path = os.path.dirname(directory)
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_path)
                zipf.write(file_path, rel_path)
    return output_filename


def upload_files_to_hf(local_path, repo_id, repo_type, target_folder=None, zip_mode=False):
    api = HfApi()

    if not os.path.exists(local_path):
        return

    # 如果需要打包成zip
    if zip_mode:
        dir_name = os.path.basename(os.path.normpath(local_path))
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
        zip_file = zip_directory(local_path, temp_zip)

        target_path = f"{dir_name}.zip"
        if target_folder:
            target_path = f"{target_folder}/{target_path}"

        api.upload_file(
            path_or_fileobj=zip_file,
            path_in_repo=target_path,
            repo_id=repo_id,
            repo_type=repo_type
        )

        os.unlink(zip_file)
        return

    files_to_upload = []
    if os.path.isfile(local_path):
        files_to_upload = [local_path]
    else:
        files_to_upload = [str(path) for path in Path(local_path).rglob('*') if path.is_file()]

    if not files_to_upload:
        return


    for file_path in files_to_upload:
        rel_path = os.path.relpath(file_path, local_path)
        if target_folder:
            target_path = f"{target_folder}/{rel_path}"
        else:
            target_path = rel_path

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=target_path,
            repo_id=repo_id,
            repo_type=repo_type
        )


def main(args):

    # token = os.environ.get("HF_TOKEN")
    # if not token:
    #     return
    #
    # login(token)

    if args.log_path:
        upload_files_to_hf(
            args.log_path,
            repo_id="NoCode-bench/Logs",
            repo_type="dataset",
            target_folder=args.target_folder,
            zip_mode=args.zip_mode
        )

    if args.trajs_path:
        upload_files_to_hf(
            args.trajs_path,
            repo_id="NoCode-bench/Trajs",
            repo_type="dataset",
            target_folder=args.target_folder,
            zip_mode=args.zip_mode
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, help="Path to Logs folder")
    parser.add_argument("--trajs_path", type=str, help="Path to Trajs folder")
    parser.add_argument("--target_folder", type=str, help="Target folder in the repository", default=None)
    parser.add_argument("--zip_mode", action="store_true", help="Upload as zip file instead of individual files")
    main(parser.parse_args())

