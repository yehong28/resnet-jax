echo 'To kill jobs in: '$VM_NAME 'in' $ZONE' after 2s...'
sleep 2s

echo 'Killing jobs...'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
pgrep -af python | grep 'main.py' | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$1}' | sh
" # &> /dev/null
echo 'Killed jobs.'

# sudo lsof -w /dev/accel0 | grep main.py | awk '{print "sudo kill -9 " $2}' | sh
# pgrep -af python | grep 'main.py' | awk '{"sudo kill -9 " $2}' | sh

