# Face Verification
Face verification system between image in documents and selfie images

## How to run
- Build your own docker images : `docker build -t face_verification:latest .`
- Or pull it from DockerHub: `docker pull lamhoangtung/face_verification`
- Then run it with: `docker run -p 8082:8082 -it face_verification:latest`

## How to use
- An API will be launched serve at `http://0.0.0.0:8082/api/predict`
- Parse image of the document to `image1` and the selfie image to `image2` key.
![](https://www.upsieutoc.com/images/2019/10/09/Screen-Shot-2019-10-09-at-1.50.27-PM.png)
- Here is the sample output:
```json
{
  "distance": 0.3416854917943924,
  "matched": "True",
  "matched_strict": "True",
  "run_time": 18.805808544158936
}
```

## Credits
Thanks **ageitgey** for the amazaing works at https://github.com/ageitgey/face_recognition
