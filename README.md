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

1. Building a Docker Image

   ```shell
   docker build -t paddleocrfastapi:latest .
   ```


2. Create the Docker container and run

   ```shell
   docker compose up -d
   ```

3. Swagger Page at `localhost:<port>/docs`