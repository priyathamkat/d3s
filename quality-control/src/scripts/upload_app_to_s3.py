import mimetypes
from pathlib import Path

import boto3
from absl import app

BUCKET_NAME = "d3s-bucket"
FOLDER_NAME = "amt_app"


def main(argv):
    # delete older build files
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(BUCKET_NAME)
    bucket.objects.filter(Prefix=FOLDER_NAME).delete()

    client = boto3.client("s3")

    build_dir = Path(__file__).parent.parent.parent / "build"
    for path in build_dir.glob("**/*"):
        if path.is_file():
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type and path.suffix == ".map":
                # both .js.map and .css.map files are not recognized by mimetypes
                # and should be set to application/json
                mime_type = "application/json"
            key = f"{FOLDER_NAME}/{path.relative_to(build_dir)}"
            client.upload_file(
                str(path), BUCKET_NAME, key, ExtraArgs={"ContentType": mime_type}
            )


if __name__ == "__main__":
    app.run(main)
