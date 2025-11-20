Implementation of mean flows for a replication study and extension of "Mean Flows for One-step Generative Modeling" (Geng et al., 2025), for the final project in the course DD2610 Deep Learning, Advanced Course at KTH


To download image net dataset onto google cloud vm instance, run the following commands in terminal:

1. SSH into your google cloud vm instance.
````gcloud compute ssh YOUR_VM_NAME --zone=YOUR_ZONE```

2. Make the setup_imagenet.sh script executable:
````chmod +x setup_imagenet.sh```

3. If kaggle.json is not already present on your VM, upload it using:
````gcloud compute scp kaggle.json YOUR_VM_NAME:~/.kaggle/kaggle.json --zone=YOUR_ZONE````

4. Run the setup_imagenet.sh script:
````./setup_imagenet.sh```


