## Setup Google Cloud Vision Account
To setup GCP account you need to:
- Create google cloud account.
- Activate billing.
- Create a project.
- Enable GCP Vision Service.
- Copy credentials on your local system in json file.

## Some Useful Links
- https://cloud.google.com/vision/docs/quickstart
- https://cloud.google.com/vision/docs/ocr

## Setup on Ubuntu-20.04
On freshly installed ubuntu system run following commands.

- Install dependencies:
`sudo apt install vim git python3-pip -y`
- Clone the code.
`git clone https://github.com/nasirkamal/GCP_VisionAPI.git`
`cd GCP_VisionAPI/`
- Install python packages.
`pip3 install -r requirement.txt`
- Run demo code.
`python3 GCVisionAPI_demo.py`

This python script `GCVisionAPI_demo.py` will detect text on image `image.jpg`, will show the result and save it as `image_detect.jpg`.

Feel free to modify the code.

# NOTE: `apikey.json` is linked to the author's Google Cloud Account which will expire shortly. To continue using please get your own account and credentials file.
