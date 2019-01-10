sudo ssh -i "/Users/justinglibert/.ssh/ec2-gpu-ireland.pem" -L 443:127.0.0.1:8888 -R 52698:127.0.0.1:52698 -L 6006:127.0.01:6006 ubuntu@ec2-63-33-178-116.eu-west-1.compute.amazonaws.com


