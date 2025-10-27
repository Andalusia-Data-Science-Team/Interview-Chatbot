@echo off
echo ===== Deploying Chatbot to Linux Server =====

scp -r "D:\New Website\Chatbot\repo\*" ai@10.24.105.221:/home/ai/repo/

ssh ai@10.24.105.221 "bash /home/ai/repo/restart.sh"

echo ===== Deployment Complete =====


:: .\deploy.bat