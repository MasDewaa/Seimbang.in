### Deploy Directly


1. (Optional) Create new virtual environment to avoid dependency conflicts
2. Install required dependencies

   ```shell
   pip3 install -r requirements.txt
   ```

3. Run FastAPI

   ```shell
   uvicorn main:app --host 0.0.0.0
   ```

### Docker Deployment

Test completed in `Centos 7`, `Ubuntu 20.04`, `Ubuntu 22.04`, `Windows 10`, `Windows 11`, requires `Docker` to be installed.

2. Building a Docker Image

   ```shell
   docker build -t paddleocrfastapi:latest .
   ```


3. Create the Docker container and run

   ```shell
   docker compose up -d
   ```

5. Swagger Page at `localhost:<port>/docs`