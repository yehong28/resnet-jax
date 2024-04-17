echo 'To kill jobs in: '$VM_NAME 'in' $ZONE' after 2s...'
sleep 2s

echo 'Killing jobs...'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
sudo pkill python
sudo lsof -w /dev/accel0 | grep main.py | awk '{print \"sudo kill -9 \" \$2}' | sh
sudo lsof -w /dev/accel0
" &> /dev/null
echo 'Killed jobs.'

# sudo lsof -w /dev/accel0 | grep main.py | awk '{print "sudo kill -9 " $2} | sh'