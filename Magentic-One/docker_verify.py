import docker

client = docker.from_env()
try:
    print(client.version())
except Exception as e:
    print("Error connecting to Docker:", e)