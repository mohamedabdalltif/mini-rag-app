# Docker Cleanup Guide: Local vs. Production

> [!CAUTION]
> **Warning:** The commands listed below are intended for **local development environments only**. Running these commands will result in a total wipe of your Docker resources.

---

## 1. The "Fresh Start" Commands
These commands are used to stop and delete all containers, images, and volumes.

```bash
# Stop all running containers
sudo docker stop $(sudo docker ps -aq)

# Remove all containers
sudo docker rm $(sudo docker ps -aq)

# Remove all images
sudo docker rmi $(sudo docker images -q)

# Remove all volumes (Deletes persistent data!)
sudo docker volume rm $(sudo docker volume ls -aq)

# Deep clean the entire system
sudo docker system prune --all